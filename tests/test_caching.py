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
Unit tests for caching functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path

import model_compass as mc
from model_compass.caching import get_config_cache, get_resolution_cache, clear_all_caches


class TestCaching:
    """Test cases for caching functionality."""
    
    def test_configuration_caching(self):
        """Test configuration file caching."""
        # Create temporary config file
        config = {
            "models": {
                "test-model": {
                    "provider": "test-provider",
                    "model": "test-model-name"
                }
            },
            "providers": {
                "test-provider": {
                    "base_url": "https://test.com"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f)
            f.flush()
            config_path = f.name
        
        try:
            # Test caching directly through the cache API
            cache = get_config_cache()
            cache.clear()
            
            # First access - should be a miss
            cached_config = cache.get(config_path)
            assert cached_config is None
            
            # Load and cache the config
            with open(config_path, 'r', encoding="utf-8") as f:
                loaded_config = json.load(f)
            cache.put(config_path, loaded_config)
            
            # Second access - should be a hit
            cached_config = cache.get(config_path)
            assert cached_config is not None
            assert cached_config == loaded_config
            
            # Check cache stats
            stats = cache.get_stats()
            assert stats["size"] == 1
            
        finally:
            Path(config_path).unlink()
    
    def test_resolution_caching(self):
        """Test resolution result caching."""
        config = {
            "models": {
                "test-model": {
                    "provider": "test-provider",
                    "model": "test-model-name"
                }
            },
            "providers": {
                "test-provider": {
                    "base_url": "https://test.com"
                }
            }
        }
        
        mc.configure(config_dict=config)
        mc.enable_performance_optimizations()
        
        # Clear resolution cache
        resolution_cache = get_resolution_cache()
        resolution_cache.invalidate()
        
        # First resolution
        model1 = mc.model("test-model")
        provider1 = model1.provider  # Trigger resolution
        
        # Second resolution - should use cache
        model2 = mc.model("test-model")
        provider2 = model2.provider  # Should be cached
        
        assert provider1 == provider2
        
        # Check cache stats
        stats = resolution_cache.get_stats()
        assert stats["size"] >= 1
    
    def test_performance_optimizations_api(self):
        """Test performance optimization API functions."""
        config = {
            "models": {
                "test-model": {
                    "provider": "test-provider",
                    "model": "test-model-name"
                }
            },
            "providers": {
                "test-provider": {
                    "base_url": "https://test.com"
                }
            }
        }
        
        mc.configure(config_dict=config)
        
        # Enable performance optimizations
        mc.enable_performance_optimizations()
        
        # Get performance stats
        stats = mc.get_performance_stats()
        
        # Verify stats structure
        assert "configuration_cache" in stats
        assert "resolution_cache" in stats
        assert "intent_scoring_cache" in stats
        assert "optimized_intent_matcher" in stats
        
        # Verify each cache has expected fields
        for cache_name, cache_stats in stats.items():
            assert isinstance(cache_stats, dict)
            if cache_name != "optimized_intent_matcher":
                assert "size" in cache_stats
                assert "hits" in cache_stats
                assert "misses" in cache_stats
    
    def test_intent_matching_optimization(self):
        """Test optimized intent matching with caching."""
        config = {
            "models": {
                "fast-model": {
                    "provider": "test",
                    "model": "fast-model",
                    "metadata": {
                        "speed": "fast",
                        "cost": "low"
                    }
                },
                "slow-model": {
                    "provider": "test",
                    "model": "slow-model",
                    "metadata": {
                        "speed": "slow",
                        "cost": "high"
                    }
                }
            },
            "providers": {
                "test": {
                    "base_url": "https://test.com"
                }
            }
        }
        
        mc.configure(config_dict=config)
        mc.enable_performance_optimizations()
        
        # First intent resolution
        model1 = mc.model("urn:llm:intent:chat?speed=fast")
        name1 = model1.name
        
        # Second intent resolution with same criteria - should use cache
        model2 = mc.model("urn:llm:intent:chat?speed=fast")
        name2 = model2.name
        
        assert name1 == name2 == "fast-model"
        
        # Check that caching is working
        stats = mc.get_performance_stats()
        intent_cache = stats["optimized_intent_matcher"]
        assert intent_cache["scoring_cache_size"] > 0
    
    def test_cache_invalidation(self):
        """Test cache invalidation when configuration changes."""
        config1 = {
            "models": {
                "model1": {
                    "provider": "provider1",
                    "model": "model1"
                }
            },
            "providers": {
                "provider1": {
                    "base_url": "https://provider1.com"
                }
            }
        }
        
        config2 = {
            "models": {
                "model2": {
                    "provider": "provider2",
                    "model": "model2"
                }
            },
            "providers": {
                "provider2": {
                    "base_url": "https://provider2.com"
                }
            }
        }
        
        # Configure with first config
        mc.configure(config_dict=config1)
        mc.enable_performance_optimizations()
        
        # Access model to populate cache
        model1 = mc.model("model1")
        provider1 = model1.provider
        
        # Switch to second config - should invalidate caches
        mc.configure(config_dict=config2)
        
        # Old model should no longer be accessible
        try:
            old_model = mc.model("model1")
            _ = old_model.provider
            pytest.fail("Expected ResolutionError for old model after config switch")
        except mc.ResolutionError:
            pass  # Expected
        
        # New model should be accessible
        model2 = mc.model("model2")
        provider2 = model2.provider
        assert provider2 == "provider2"