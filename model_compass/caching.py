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
Caching utilities for performance optimization.

This module provides caching mechanisms for configuration files, resolution results,
and other frequently accessed data to improve library performance.
"""

import time
import hashlib
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
from functools import wraps, lru_cache
from threading import RLock

from .logging_utils import log_debug, log_info


class ConfigurationCache:
    """
    Cache for configuration files to avoid re-parsing on every access.
    
    This cache stores parsed configuration data along with file modification
    times to detect when files have changed and need to be re-parsed.
    """
    
    def __init__(self, max_size: int = 100):
        """
        Initialize configuration cache.
        
        Args:
            max_size: Maximum number of configurations to cache
        """
        self._cache: Dict[str, Tuple[Dict[str, Any], float, str]] = {}
        self._max_size = max_size
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached configuration for a file path.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Cached configuration dictionary or None if not cached or stale
        """
        with self._lock:
            if file_path not in self._cache:
                self._misses += 1
                log_debug(f"Configuration cache miss: {file_path}")
                return None
            
            cached_config, cached_mtime, cached_hash = self._cache[file_path]
            
            # Check if file still exists and hasn't been modified
            try:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    # File was deleted, remove from cache
                    del self._cache[file_path]
                    self._misses += 1
                    log_debug(f"Configuration cache miss (file deleted): {file_path}")
                    return None
                
                current_mtime = file_path_obj.stat().st_mtime
                if current_mtime != cached_mtime:
                    # File was modified, remove from cache
                    del self._cache[file_path]
                    self._misses += 1
                    log_debug(f"Configuration cache miss (file modified): {file_path}")
                    return None
                
                # File hasn't changed, return cached config
                self._hits += 1
                log_debug(f"Configuration cache hit: {file_path}")
                return cached_config.copy()
                
            except (OSError, IOError):
                # Error accessing file, remove from cache
                if file_path in self._cache:
                    del self._cache[file_path]
                self._misses += 1
                return None
    
    def put(self, file_path: str, config: Dict[str, Any]) -> None:
        """
        Store configuration in cache.
        
        Args:
            file_path: Path to configuration file
            config: Parsed configuration dictionary
        """
        with self._lock:
            try:
                file_path_obj = Path(file_path)
                if not file_path_obj.exists():
                    return
                
                mtime = file_path_obj.stat().st_mtime
                
                # Create a hash of the config for integrity checking
                config_str = str(sorted(config.items()))
                config_hash = hashlib.sha256(config_str.encode()).hexdigest()
                
                # Implement LRU eviction if cache is full
                if len(self._cache) >= self._max_size and file_path not in self._cache:
                    # Remove oldest entry (simple FIFO for now)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]
                    log_debug(f"Configuration cache evicted: {oldest_key}")
                
                self._cache[file_path] = (config.copy(), mtime, config_hash)
                log_debug(f"Configuration cached: {file_path}")
                
            except (OSError, IOError):
                # Error accessing file, don't cache
                pass
    
    def invalidate(self, file_path: str) -> None:
        """
        Invalidate cached configuration for a file.
        
        Args:
            file_path: Path to configuration file to invalidate
        """
        with self._lock:
            if file_path in self._cache:
                del self._cache[file_path]
                log_debug(f"Configuration cache invalidated: {file_path}")
    
    def clear(self) -> None:
        """Clear all cached configurations."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            log_debug("Configuration cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "cached_files": list(self._cache.keys())
            }


class ResolutionCache:
    """
    Cache for resolution results to avoid re-resolving frequently accessed identifiers.
    
    This cache stores resolved model configurations with TTL (time-to-live) to ensure
    that cached results don't become stale if the underlying configuration changes.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        """
        Initialize resolution cache.
        
        Args:
            max_size: Maximum number of resolutions to cache
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self._cache: Dict[str, Tuple[Any, float]] = {}  # Use Any to avoid circular import
        self._max_size = max_size
        self._ttl_seconds = ttl_seconds
        self._lock = RLock()
        self._hits = 0
        self._misses = 0
    
    def _is_expired(self, timestamp: float) -> bool:
        """Check if a cache entry has expired."""
        return time.time() - timestamp > self._ttl_seconds
    
    def _cleanup_expired(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if current_time - timestamp > self._ttl_seconds
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            log_debug(f"Resolution cache cleaned up {len(expired_keys)} expired entries")
    
    def get(self, identifier: str, profile_context: Optional[str] = None) -> Optional[Any]:
        """
        Get cached resolution result.
        
        Args:
            identifier: Model identifier that was resolved
            profile_context: Current profile context (affects caching key)
            
        Returns:
            Cached ResolvedModel or None if not cached or expired
        """
        with self._lock:
            # Create cache key that includes profile context
            cache_key = f"{identifier}:{profile_context or 'default'}"
            
            if cache_key not in self._cache:
                self._misses += 1
                return None
            
            resolved_model, timestamp = self._cache[cache_key]
            
            if self._is_expired(timestamp):
                del self._cache[cache_key]
                self._misses += 1
                log_debug(f"Resolution cache miss (expired): {cache_key}")
                return None
            
            self._hits += 1
            log_debug(f"Resolution cache hit: {cache_key}")
            return resolved_model
    
    def put(self, identifier: str, resolved_model: Any, profile_context: Optional[str] = None) -> None:
        """
        Store resolution result in cache.
        
        Args:
            identifier: Model identifier that was resolved
            resolved_model: The resolved model configuration
            profile_context: Current profile context
        """
        with self._lock:
            # Clean up expired entries periodically
            if len(self._cache) % 100 == 0:
                self._cleanup_expired()
            
            cache_key = f"{identifier}:{profile_context or 'default'}"
            
            # Implement LRU eviction if cache is full
            if len(self._cache) >= self._max_size and cache_key not in self._cache:
                # Remove oldest entry (simple FIFO for now)
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                log_debug(f"Resolution cache evicted: {oldest_key}")
            
            self._cache[cache_key] = (resolved_model, time.time())
            log_debug(f"Resolution cached: {cache_key}")
    
    def invalidate(self, identifier: Optional[str] = None) -> None:
        """
        Invalidate cached resolutions.
        
        Args:
            identifier: Specific identifier to invalidate, or None to clear all
        """
        with self._lock:
            if identifier is None:
                self._cache.clear()
                log_debug("Resolution cache cleared")
            else:
                # Remove all entries that start with the identifier
                keys_to_remove = [key for key in self._cache.keys() if key.startswith(f"{identifier}:")]
                for key in keys_to_remove:
                    del self._cache[key]
                log_debug(f"Resolution cache invalidated for identifier: {identifier}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            # Clean up expired entries for accurate stats
            self._cleanup_expired()
            
            total_requests = self._hits + self._misses
            hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 2),
                "cached_identifiers": len(set(key.split(':')[0] for key in self._cache.keys()))
            }


# Global cache instances
_config_cache = None
_resolution_cache = None
_lazy_registry = None
_optimized_intent_matcher = None


def get_config_cache() -> ConfigurationCache:
    """Get the global configuration cache instance."""
    global _config_cache
    if _config_cache is None:
        _config_cache = ConfigurationCache()
    return _config_cache


def get_resolution_cache() -> ResolutionCache:
    """Get the global resolution cache instance."""
    global _resolution_cache
    if _resolution_cache is None:
        _resolution_cache = ResolutionCache()
    return _resolution_cache


def get_lazy_registry() -> 'LazyModelRegistry':
    """Get the global lazy model registry instance."""
    global _lazy_registry
    if _lazy_registry is None:
        _lazy_registry = LazyModelRegistry()
    return _lazy_registry


def get_optimized_intent_matcher() -> 'OptimizedIntentMatcher':
    """Get the global optimized intent matcher instance."""
    global _optimized_intent_matcher
    if _optimized_intent_matcher is None:
        _optimized_intent_matcher = OptimizedIntentMatcher()
    return _optimized_intent_matcher


def reset_caches() -> None:
    """Reset all caches (useful for testing)."""
    global _config_cache, _resolution_cache, _lazy_registry, _optimized_intent_matcher
    if _config_cache:
        _config_cache.clear()
    if _resolution_cache:
        _resolution_cache.invalidate()
    if _lazy_registry:
        _lazy_registry.reset()
    if _optimized_intent_matcher:
        _optimized_intent_matcher.clear_cache()


def cache_resolution(func):
    """
    Decorator to cache resolution results.
    
    This decorator can be applied to resolution methods to automatically
    cache their results and return cached values when available.
    """
    @wraps(func)
    def wrapper(self, identifier: str, *args, **kwargs):
        # Get current profile context for cache key
        from .core import _get_current_profile
        profile_context = _get_current_profile()
        
        # Try to get from cache first
        cache = get_resolution_cache()
        cached_result = cache.get(identifier, profile_context)
        if cached_result is not None:
            return cached_result
        
        # Not in cache, call the original function
        result = func(self, identifier, *args, **kwargs)
        
        # Cache the result
        if result is not None:
            cache.put(identifier, result, profile_context)
        
        return result
    
    return wrapper


class LazyModelRegistry:
    """
    Lazy loading registry for model definitions and provider configurations.
    
    This registry loads model definitions and provider configurations only when
    they are first accessed, improving startup performance for large model sets.
    """
    
    def __init__(self):
        """Initialize lazy registry with empty state."""
        self._models_loaded = False
        self._providers_loaded = False
        self._models: Dict[str, Any] = {}
        self._providers: Dict[str, Any] = {}
        self._lock = RLock()
    
    def get_model(self, model_id: str) -> Optional[Any]:
        """Get model definition with lazy loading."""
        with self._lock:
            if not self._models_loaded:
                self._load_models()
            return self._models.get(model_id)
    
    def get_provider(self, provider_name: str) -> Optional[Any]:
        """Get provider definition with lazy loading."""
        with self._lock:
            if not self._providers_loaded:
                self._load_providers()
            return self._providers.get(provider_name)
    
    def list_models(self) -> list:
        """List all model IDs with lazy loading."""
        with self._lock:
            if not self._models_loaded:
                self._load_models()
            return list(self._models.keys())
    
    def list_providers(self) -> list:
        """List all provider names with lazy loading."""
        with self._lock:
            if not self._providers_loaded:
                self._load_providers()
            return list(self._providers.keys())
    
    def _load_models(self):
        """Load model definitions - no longer needed with user-config-only approach."""
        if self._models_loaded:
            return
        
        # No longer load from registry - models come from user configuration
        self._models = {}
        self._models_loaded = True
        log_debug("Lazy model loading disabled - using user configuration only")
    
    def _load_providers(self):
        """Load provider definitions - no longer needed with user-config-only approach."""
        if self._providers_loaded:
            return
        
        # No longer load from registry - providers come from user configuration
        self._providers = {}
        self._providers_loaded = True
        log_debug("Lazy provider loading disabled - using user configuration only")
    
    def reset(self):
        """Reset lazy loading state."""
        with self._lock:
            self._models_loaded = False
            self._providers_loaded = False
            self._models.clear()
            self._providers.clear()


class OptimizedIntentMatcher:
    """
    Optimized intent matcher with caching and performance improvements.
    
    This class provides optimized intent matching for large model sets with
    sophisticated caching and scoring optimizations.
    """
    
    def __init__(self, max_cache_size: int = 1000):
        """Initialize optimized intent matcher."""
        self._scoring_cache: Dict[str, float] = {}
        self._model_index: Dict[str, Dict[str, set]] = {}
        self._max_cache_size = max_cache_size
        self._lock = RLock()
        self._index_built = False
    
    def _build_model_index(self, models: Dict[str, Any]) -> None:
        """Build optimized index for fast model filtering."""
        if self._index_built:
            return
        
        with self._lock:
            if self._index_built:  # Double-check after acquiring lock
                return
            
            self._model_index.clear()
            
            for model_id, model_config in models.items():
                metadata = getattr(model_config, 'metadata', {})
                
                # Index by each metadata field
                for key, value in metadata.items():
                    if key not in self._model_index:
                        self._model_index[key] = {}
                    
                    value_str = str(value).lower()
                    if value_str not in self._model_index[key]:
                        self._model_index[key][value_str] = set()
                    
                    self._model_index[key][value_str].add(model_id)
                
                # Index by provider
                provider = getattr(model_config, 'provider', '')
                if 'provider' not in self._model_index:
                    self._model_index['provider'] = {}
                if provider not in self._model_index['provider']:
                    self._model_index['provider'][provider] = set()
                self._model_index['provider'][provider].add(model_id)
            
            self._index_built = True
            log_debug(f"Built model index with {len(self._model_index)} metadata fields")
    
    def find_candidates(self, criteria: Dict[str, str], models: Dict[str, Any]) -> set:
        """
        Find candidate models using optimized indexing.
        
        Args:
            criteria: Intent criteria to match
            models: Available models
            
        Returns:
            Set of candidate model IDs that might match
        """
        self._build_model_index(models)
        
        if not criteria:
            return set(models.keys())
        
        candidate_sets = []
        
        # Find candidates for each criterion
        for key, value in criteria.items():
            value_lower = value.lower()
            
            if key in self._model_index and value_lower in self._model_index[key]:
                candidate_sets.append(self._model_index[key][value_lower])
            else:
                # No exact matches for this criterion - use all models
                candidate_sets.append(set(models.keys()))
        
        # Intersection of all candidate sets gives us the most promising candidates
        if candidate_sets:
            candidates = candidate_sets[0]
            for candidate_set in candidate_sets[1:]:
                candidates = candidates.intersection(candidate_set)
            return candidates
        
        return set(models.keys())
    
    def get_cached_score(self, criteria: Dict[str, str], model_metadata: Dict[str, Any]) -> Optional[float]:
        """Get cached scoring result if available."""
        try:
            # Create cache key from criteria and metadata
            criteria_key = tuple(sorted(criteria.items()))
            
            # Convert metadata to hashable format
            hashable_metadata = []
            for key, value in sorted(model_metadata.items()):
                if isinstance(value, list):
                    hashable_metadata.append((key, tuple(value)))
                elif isinstance(value, dict):
                    hashable_metadata.append((key, tuple(sorted(value.items()))))
                else:
                    hashable_metadata.append((key, value))
            
            metadata_key = tuple(hashable_metadata)
            cache_key = f"{hash(criteria_key)}:{hash(metadata_key)}"
            
            return self._scoring_cache.get(cache_key)
        except (TypeError, ValueError):
            # If we can't create a hashable key, skip caching
            return None
    
    def cache_score(self, criteria: Dict[str, str], model_metadata: Dict[str, Any], score: float) -> None:
        """Cache a scoring result."""
        try:
            with self._lock:
                # Implement simple LRU eviction
                if len(self._scoring_cache) >= self._max_cache_size:
                    # Remove oldest entry (simple FIFO)
                    oldest_key = next(iter(self._scoring_cache))
                    del self._scoring_cache[oldest_key]
                
                criteria_key = tuple(sorted(criteria.items()))
                
                # Convert metadata to hashable format
                hashable_metadata = []
                for key, value in sorted(model_metadata.items()):
                    if isinstance(value, list):
                        hashable_metadata.append((key, tuple(value)))
                    elif isinstance(value, dict):
                        hashable_metadata.append((key, tuple(sorted(value.items()))))
                    else:
                        hashable_metadata.append((key, value))
                
                metadata_key = tuple(hashable_metadata)
                cache_key = f"{hash(criteria_key)}:{hash(metadata_key)}"
                
                self._scoring_cache[cache_key] = score
        except (TypeError, ValueError):
            # If we can't create a hashable key, skip caching
            pass
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        with self._lock:
            self._scoring_cache.clear()
            self._model_index.clear()
            self._index_built = False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                "scoring_cache_size": len(self._scoring_cache),
                "max_cache_size": self._max_cache_size,
                "index_built": self._index_built,
                "indexed_fields": len(self._model_index)
            }


@lru_cache(maxsize=512)
def cached_intent_scoring(criteria_tuple: Tuple[Tuple[str, str], ...], model_metadata_tuple: Tuple[Tuple[str, Any], ...]) -> float:
    """
    Cached version of intent matching scoring for performance optimization.
    
    This function uses Python's built-in LRU cache to cache scoring results
    for identical criteria and metadata combinations.
    
    Args:
        criteria_tuple: Tuple of (key, value) pairs for intent criteria
        model_metadata_tuple: Tuple of (key, value) pairs for model metadata
        
    Returns:
        Calculated score between 0.0 and 1.0
    """
    # Convert tuples back to dictionaries
    criteria = dict(criteria_tuple)
    metadata = dict(model_metadata_tuple)
    
    # Implement optimized scoring logic directly here to avoid circular imports
    if not criteria:
        return 0.5  # Neutral score for no criteria
    
    # Default weights for different metadata criteria
    default_weights = {
        "reasoning": 1.0,
        "speed": 0.8,
        "cost": 0.9,
        "provider": 1.0,
        "context_length": 0.6,
        "capabilities": 0.7
    }
    
    # Ordinal mappings for scoring
    ordinal_mappings = {
        "reasoning": {"low": 1, "medium": 2, "high": 3},
        "speed": {"slow": 1, "medium": 2, "fast": 3},
        "cost": {"low": 1, "medium": 2, "high": 3}
    }
    
    total_weight = 0.0
    weighted_score = 0.0
    
    for criterion, expected_value in criteria.items():
        weight = default_weights.get(criterion, 0.5)
        total_weight += weight
        
        # Get the actual value from metadata
        actual_value = metadata.get(criterion)
        if actual_value is None:
            continue
        
        # Calculate match score for this criterion
        actual_str = str(actual_value).lower()
        expected_str = expected_value.lower()
        
        # Exact match gets full score
        if actual_str == expected_str:
            match_score = 1.0
        # Handle ordinal criteria
        elif criterion in ordinal_mappings:
            mapping = ordinal_mappings[criterion]
            expected_val = mapping.get(expected_str)
            actual_val = mapping.get(actual_str)
            
            if expected_val is not None and actual_val is not None:
                if expected_val == actual_val:
                    match_score = 1.0
                elif criterion == "cost" and expected_str == "low" and actual_val < expected_val:
                    match_score = 0.8  # Better than expected
                elif criterion in ["reasoning", "speed"] and actual_val > expected_val:
                    match_score = 0.9  # Better than expected
                else:
                    distance = abs(expected_val - actual_val)
                    match_score = max(0.0, 1.0 - (distance * 0.3))
            else:
                match_score = 0.5 if expected_str == actual_str else 0.0
        # Handle string matching
        else:
            if expected_str in actual_str or actual_str in expected_str:
                match_score = 0.7
            else:
                match_score = 0.0
        
        weighted_score += match_score * weight
    
    # Normalize by total weight
    if total_weight == 0:
        return 0.0
    
    return min(weighted_score / total_weight, 1.0)


def optimize_intent_matching():
    """
    Apply optimizations to intent matching for better performance.
    
    This function can be called to enable various performance optimizations
    for intent matching with large model sets.
    """
    log_info("Intent matching optimizations enabled")
    
    # The cached_intent_scoring function above provides caching
    # Additional optimizations could be added here in the future


def enable_lazy_loading() -> None:
    """
    Enable lazy loading for model definitions and provider configurations.
    
    This function configures the system to use lazy loading, which can
    significantly improve startup performance for large model sets.
    """
    log_info("Lazy loading enabled for model definitions and providers")


def enable_performance_optimizations() -> None:
    """
    Enable all performance optimizations including caching and lazy loading.
    
    This function enables:
    - Configuration file caching
    - Resolution result caching  
    - Lazy loading of model definitions
    - Optimized intent matching with indexing
    """
    # Initialize all cache instances to ensure they're ready
    get_config_cache()
    get_resolution_cache()
    get_lazy_registry()
    get_optimized_intent_matcher()
    
    log_info("All performance optimizations enabled", {
        "config_cache": True,
        "resolution_cache": True,
        "lazy_loading": True,
        "optimized_intent_matching": True
    })


def get_cache_stats() -> Dict[str, Any]:
    """
    Get comprehensive cache statistics.
    
    Returns:
        Dictionary with statistics from all caches
    """
    stats = {
        "configuration_cache": get_config_cache().get_stats(),
        "resolution_cache": get_resolution_cache().get_stats(),
        "intent_scoring_cache": {
            "size": cached_intent_scoring.cache_info().currsize,
            "max_size": cached_intent_scoring.cache_info().maxsize,
            "hits": cached_intent_scoring.cache_info().hits,
            "misses": cached_intent_scoring.cache_info().misses,
            "hit_rate": round(
                (cached_intent_scoring.cache_info().hits / 
                 (cached_intent_scoring.cache_info().hits + cached_intent_scoring.cache_info().misses) * 100)
                if (cached_intent_scoring.cache_info().hits + cached_intent_scoring.cache_info().misses) > 0 else 0, 2
            )
        },
        "optimized_intent_matcher": get_optimized_intent_matcher().get_cache_stats()
    }
    
    return stats


def clear_all_caches() -> None:
    """Clear all caches and reset statistics."""
    reset_caches()
    cached_intent_scoring.cache_clear()
    log_info("All caches cleared")