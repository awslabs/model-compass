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
Resolution engine for resolving model identifiers to configurations.

This module handles the core logic for resolving different types of model 
identifiers to concrete configurations, including physical identifiers,
profile-based resolution, alias chains, and intent-based resolution.
"""

import re
from typing import Optional, Dict, Any, List, Tuple
from urllib.parse import urlparse, parse_qs

from .data_models import ResolvedModel, ModelConfig, ProviderConfig
from .exceptions import ResolutionError, CircularAliasError
from .logging_utils import (
    log_debug, log_warning, log_resolution_attempt, log_resolution_success,
    log_alias_chain_resolution, log_intent_matching, create_error_context
)
from .caching import cache_resolution, get_resolution_cache, cached_intent_scoring


class AliasResolver:
    """
    Handles alias chain resolution with circular reference detection.
    
    This class is responsible for resolving alias chains while preventing
    infinite loops through circular reference detection. It provides methods
    to resolve aliases to their final targets and to show the full resolution
    path for debugging purposes.
    """
    
    def __init__(self, configuration_manager):
        """
        Initialize alias resolver with configuration manager.
        
        Args:
            configuration_manager: ConfigurationManager instance
        """
        self._config_manager = configuration_manager
    
    def resolve_alias(self, alias: str) -> str:
        """
        Resolve alias chain to final target identifier.
        
        Follows alias chains while detecting circular references to prevent
        infinite loops. Returns the final non-alias target.
        
        Args:
            alias: Alias name to resolve
            
        Returns:
            Final target identifier (model ID, profile name, or physical identifier)
            
        Raises:
            ResolutionError: If alias cannot be resolved
            CircularAliasError: If circular reference detected
        """
        resolution_chain = []
        return self._resolve_alias_recursive(alias, resolution_chain)
    
    def resolve_chain(self, alias: str) -> List[str]:
        """
        Show the full resolution path for an alias for debugging.
        
        Returns the complete chain from the initial alias to the final target,
        including all intermediate aliases.
        
        Args:
            alias: Alias name to trace
            
        Returns:
            List of identifiers showing the full resolution path
            
        Raises:
            ResolutionError: If alias cannot be resolved
            CircularAliasError: If circular reference detected
        """
        resolution_chain = []
        final_target = self._resolve_alias_with_chain_tracking(alias, resolution_chain)
        
        # Return the complete chain including the final target
        return resolution_chain + [final_target]
    
    def _resolve_alias_with_chain_tracking(self, current_identifier: str, resolution_chain: List[str]) -> str:
        """
        Internal method to resolve alias while tracking the full chain.
        
        This method is similar to _resolve_alias_recursive but tracks all
        identifiers in the chain, not just aliases.
        
        Args:
            current_identifier: Current identifier being resolved
            resolution_chain: Current resolution chain being built
            
        Returns:
            Final target identifier
            
        Raises:
            CircularAliasError: If circular reference detected
            ResolutionError: If alias cannot be resolved
        """
        # Check for circular reference
        if current_identifier in resolution_chain:
            circular_chain = resolution_chain + [current_identifier]
            raise CircularAliasError(
                f"Circular alias reference detected: {' -> '.join(circular_chain)}",
                alias_chain=circular_chain
            )
        
        # Add current identifier to the chain
        resolution_chain.append(current_identifier)
        
        # If current identifier is not an alias, it's our final target
        if not self._config_manager.has_alias(current_identifier):
            # Remove the final target from the chain since we'll add it separately
            resolution_chain.pop()
            return current_identifier
        
        # Get alias target
        alias_target = self._config_manager.get_alias(current_identifier)
        if not alias_target:
            raise ResolutionError(
                f"Alias '{current_identifier}' not found",
                identifier=current_identifier,
                resolution_path=resolution_chain,
                context={
                    "available_aliases": list(self._config_manager.get_aliases().keys())
                }
            )
        
        # Continue recursively with the target
        return self._resolve_alias_with_chain_tracking(alias_target, resolution_chain)
    
    def _resolve_alias_recursive(self, current_identifier: str, resolution_chain: List[str]) -> str:
        """
        Internal recursive method to resolve alias with circular detection.
        
        Args:
            current_identifier: Current identifier being resolved
            resolution_chain: Current resolution chain for circular detection
            
        Returns:
            Final target identifier
            
        Raises:
            CircularAliasError: If circular reference detected
            ResolutionError: If alias cannot be resolved
        """
        # Check for circular reference
        if current_identifier in resolution_chain:
            circular_chain = resolution_chain + [current_identifier]
            log_warning(f"Circular alias reference detected: {' -> '.join(circular_chain)}")
            
            suggestions = [
                "Check your alias definitions for circular references",
                f"The alias '{current_identifier}' eventually points back to itself",
                "Break the circular chain by pointing one alias to a concrete model or profile",
                "Review the alias chain to identify the problematic reference"
            ]
            
            raise CircularAliasError(
                f"Circular alias reference detected: {' -> '.join(circular_chain)}",
                alias_chain=circular_chain,
                suggestions=suggestions
            )
        
        # If current identifier is not an alias, it's our final target
        if not self._config_manager.has_alias(current_identifier):
            return current_identifier
        
        # Get alias target
        alias_target = self._config_manager.get_alias(current_identifier)
        if not alias_target:
            raise ResolutionError(
                f"Alias '{current_identifier}' not found",
                identifier=current_identifier,
                resolution_path=resolution_chain + [current_identifier],
                context={
                    "available_aliases": list(self._config_manager.get_aliases().keys())
                }
            )
        
        # Add current identifier to resolution chain and continue recursively
        new_chain = resolution_chain + [current_identifier]
        return self._resolve_alias_recursive(alias_target, new_chain)
    
    def validate_alias_references(self) -> List[str]:
        """
        Validate all alias references in the configuration.
        
        Checks all aliases for circular references and invalid targets.
        This method can be used during configuration validation to detect
        issues before they cause runtime errors.
        
        Returns:
            List of validation error messages (empty if all aliases are valid)
        """
        errors = []
        aliases = self._config_manager.get_aliases()
        
        for alias_name in aliases:
            try:
                self.resolve_alias(alias_name)
            except CircularAliasError as e:
                errors.append(f"Circular reference in alias '{alias_name}': {e.message}")
            except ResolutionError as e:
                errors.append(f"Invalid alias '{alias_name}': {e.message}")
            except Exception as e:
                errors.append(f"Unexpected error validating alias '{alias_name}': {str(e)}")
        
        return errors
    
    def get_alias_dependencies(self, alias: str) -> List[str]:
        """
        Get all aliases that depend on the given alias.
        
        This method finds all aliases that directly or indirectly reference
        the given alias, which is useful for understanding the impact of
        changing an alias target.
        
        Args:
            alias: Alias name to find dependencies for
            
        Returns:
            List of alias names that depend on the given alias
        """
        dependencies = []
        aliases = self._config_manager.get_aliases()
        
        for alias_name in aliases:
            try:
                chain = self.resolve_chain(alias_name)
                if alias in chain[:-1]:  # Exclude the final target
                    dependencies.append(alias_name)
            except (CircularAliasError, ResolutionError):
                # Skip aliases that can't be resolved
                continue
        
        return dependencies


class URNParser:
    """
    Parser for URN-based intent strings.
    
    Handles parsing of URN strings in the format:
    urn:llm:intent:service?param1=value1&param2=value2
    """
    
    # URN pattern: urn:llm:intent:service?param1=value1&param2=value2
    URN_PATTERN = re.compile(r'^urn:llm:intent:([a-zA-Z0-9_-]+)(?:\?(.+))?$')
    
    @classmethod
    def parse(cls, urn: str) -> Tuple[str, Dict[str, str]]:
        """
        Parse URN string into service and parameters.
        
        Args:
            urn: URN string to parse
            
        Returns:
            Tuple of (service, parameters_dict)
            
        Raises:
            ResolutionError: If URN format is invalid
        """
        if not cls.is_valid_urn(urn):
            suggestions = [
                "Use format: urn:llm:intent:service?param1=value1&param2=value2",
                "Example: urn:llm:intent:chat?reasoning=high&cost=low",
                "Example: urn:llm:intent:completion?speed=fast",
                "Example: urn:llm:intent:chat?provider=openai"
            ]
            
            raise ResolutionError(
                f"Invalid URN format: '{urn}'. Expected 'urn:llm:intent:service?param=value'",
                identifier=urn,
                suggestions=suggestions,
                context={
                    "expected_format": "urn:llm:intent:service?param1=value1&param2=value2",
                    "examples": [
                        "urn:llm:intent:chat?reasoning=high&cost=low",
                        "urn:llm:intent:completion?speed=fast",
                        "urn:llm:intent:chat?provider=openai"
                    ]
                }
            )
        
        match = cls.URN_PATTERN.match(urn)
        service = match.group(1)
        query_string = match.group(2)
        
        # Parse query parameters
        parameters = {}
        if query_string:
            try:
                # Parse query string manually to handle simple cases
                for param_pair in query_string.split('&'):
                    if '=' in param_pair:
                        key, value = param_pair.split('=', 1)
                        # URL decode key and value
                        key = key.strip()
                        value = value.strip()
                        if key and value:
                            parameters[key] = value
            except Exception as e:
                raise ResolutionError(
                    f"Invalid query parameters in URN: '{urn}'",
                    identifier=urn,
                    context={
                        "parse_error": str(e),
                        "query_string": query_string
                    }
                )
        
        return service, parameters
    
    @classmethod
    def is_valid_urn(cls, urn: str) -> bool:
        """Check if string is a valid URN format."""
        return bool(cls.URN_PATTERN.match(urn))


class IntentMatcher:
    """
    Matches URN-based intent queries to available models using metadata scoring.
    
    This class implements a sophisticated scoring algorithm that evaluates models
    based on how well they match the intent criteria, with support for exact
    matches, partial matches, and weighted scoring.
    """
    
    def __init__(self, configuration_manager):
        """
        Initialize intent matcher with configuration manager.
        
        Args:
            configuration_manager: ConfigurationManager instance
        """
        self._config_manager = configuration_manager
        
        # Default weights for different metadata criteria
        # Higher weights mean more important for matching
        self._default_weights = {
            "reasoning": 1.0,
            "speed": 0.8,
            "cost": 0.9,
            "provider": 1.0,
            "context_length": 0.6,
            "capabilities": 0.7
        }
        
        # Value mappings for ordinal criteria (low < medium < high)
        self._ordinal_mappings = {
            "reasoning": {"low": 1, "medium": 2, "high": 3},
            "speed": {"slow": 1, "medium": 2, "fast": 3},
            "cost": {"low": 1, "medium": 2, "high": 3}
        }
    
    def find_best_match(self, service: str, criteria: Dict[str, str]) -> str:
        """
        Find the best matching model for the given intent criteria with optimization.
        
        Args:
            service: Service type (e.g., 'chat', 'completion')
            criteria: Dictionary of intent criteria
            
        Returns:
            Model ID of the best matching model
            
        Raises:
            ResolutionError: If no suitable model found
        """
        models = self._config_manager.get_models()
        
        if not models:
            raise ResolutionError(
                "No models available for intent matching",
                context={
                    "service": service,
                    "criteria": criteria
                }
            )
        
        # Use optimized candidate filtering for large model sets
        from .caching import get_optimized_intent_matcher
        optimizer = get_optimized_intent_matcher()
        
        if len(models) > 10:  # Use optimization for larger model sets
            candidates = optimizer.find_candidates(criteria, models)
            log_debug(f"Intent matching: filtered {len(models)} models to {len(candidates)} candidates")
        else:
            candidates = set(models.keys())
        
        # Score candidate models against the criteria
        scored_models = []
        for model_id in candidates:
            model_config = models[model_id]
            score = self._score_model(model_config, criteria)
            if score > 0:  # Only include models with positive scores
                scored_models.append((model_id, model_config, score))
        
        if not scored_models:
            # No models matched - provide helpful suggestions with configuration examples
            from .config_templates import ConfigurationExamples
            
            suggestions = self._generate_suggestions(criteria, models)
            available_models = list(models.keys())
            
            # Generate configuration example for intent resolution
            configuration_examples = [ConfigurationExamples.get_intent_resolution_example(criteria)]
            
            raise ResolutionError(
                f"No models found matching intent criteria: {criteria}",
                identifier=f"urn:llm:intent:{service}?" + "&".join(f"{k}={v}" for k, v in criteria.items()),
                available_models=available_models,
                suggestions=suggestions,
                configuration_examples=configuration_examples
            )
        
        # Sort by score (highest first) and return the best match
        scored_models.sort(key=lambda x: x[2], reverse=True)
        best_model_id, best_model_config, best_score = scored_models[0]
        
        log_debug(f"Intent matching: selected {best_model_id} with score {best_score:.3f}")
        return best_model_id
    
    def find_all_matches(self, service: str, criteria: Dict[str, str], min_score: float = 0.1) -> List[Tuple[str, float]]:
        """
        Find all models that match the intent criteria above a minimum score.
        
        Args:
            service: Service type (e.g., 'chat', 'completion')
            criteria: Dictionary of intent criteria
            min_score: Minimum score threshold (0.0 to 1.0)
            
        Returns:
            List of (model_id, score) tuples sorted by score (highest first)
        """
        models = self._config_manager.get_models()
        scored_models = []
        
        for model_id, model_config in models.items():
            score = self._score_model(model_config, criteria)
            if score >= min_score:
                scored_models.append((model_id, score))
        
        # Sort by score (highest first)
        scored_models.sort(key=lambda x: x[1], reverse=True)
        return scored_models
    
    def _score_model(self, model_config: ModelConfig, criteria: Dict[str, str]) -> float:
        """
        Score a model against the given criteria with caching optimization.
        
        Args:
            model_config: Model configuration to score
            criteria: Intent criteria to match against
            
        Returns:
            Score between 0.0 and 1.0 (higher is better)
        """
        if not criteria:
            return 0.5  # Neutral score for no criteria
        
        # Try to get cached score first
        from .caching import get_optimized_intent_matcher
        optimizer = get_optimized_intent_matcher()
        
        # Build metadata dict for caching
        metadata = model_config.metadata.copy()
        metadata['provider'] = model_config.provider
        metadata['model'] = model_config.model
        
        cached_score = optimizer.get_cached_score(criteria, metadata)
        if cached_score is not None:
            return cached_score
        
        # Calculate score using the original logic
        total_weight = 0.0
        weighted_score = 0.0
        
        for criterion, expected_value in criteria.items():
            weight = self._default_weights.get(criterion, 0.5)
            total_weight += weight
            
            # Get the actual value - check both metadata and model config fields
            actual_value = None
            
            if criterion == "provider":
                # Provider is a direct field in ModelConfig
                actual_value = model_config.provider
            elif criterion == "model":
                # Model name is a direct field in ModelConfig
                actual_value = model_config.model
            else:
                # Other criteria come from metadata
                actual_value = model_config.metadata.get(criterion)
            
            if actual_value is None:
                # Model doesn't have this criterion - neutral score
                continue
            
            # Calculate match score for this criterion
            match_score = self._calculate_match_score(criterion, expected_value, actual_value)
            weighted_score += match_score * weight
        
        # Normalize by total weight
        if total_weight == 0:
            score = 0.0
        else:
            score = min(weighted_score / total_weight, 1.0)
        
        # Cache the result
        optimizer.cache_score(criteria, metadata, score)
        
        return score
    
    def _calculate_match_score(self, criterion: str, expected: str, actual: Any) -> float:
        """
        Calculate match score for a specific criterion.
        
        Args:
            criterion: Name of the criterion
            expected: Expected value from intent
            actual: Actual value from model metadata
            
        Returns:
            Match score between 0.0 and 1.0
        """
        # Convert actual value to string for comparison
        actual_str = str(actual).lower()
        expected_str = expected.lower()
        
        # Exact match gets full score
        if actual_str == expected_str:
            return 1.0
        
        # Handle ordinal criteria (low/medium/high)
        if criterion in self._ordinal_mappings:
            return self._score_ordinal_match(criterion, expected_str, actual_str)
        
        # Handle numeric criteria
        if self._is_numeric(expected) and self._is_numeric(actual):
            return self._score_numeric_match(expected, actual)
        
        # Handle string criteria with partial matching
        return self._score_string_match(expected_str, actual_str)
    
    def _score_ordinal_match(self, criterion: str, expected: str, actual: str) -> float:
        """
        Score ordinal criteria (low/medium/high) with preference logic.
        
        For cost: lower is better when requesting low cost
        For reasoning/speed: higher is better when requesting high performance
        """
        mapping = self._ordinal_mappings[criterion]
        
        expected_val = mapping.get(expected)
        actual_val = mapping.get(actual)
        
        if expected_val is None or actual_val is None:
            # Fallback to string matching if not in ordinal mapping
            return 0.5 if expected == actual else 0.0
        
        # Exact match
        if expected_val == actual_val:
            return 1.0
        
        # For cost, lower actual cost is better when requesting low cost
        if criterion == "cost":
            if expected == "low" and actual_val < expected_val:
                return 0.8  # Better than expected
            elif expected == "high" and actual_val > expected_val:
                return 0.8  # Better than expected
            else:
                # Calculate distance penalty
                distance = abs(expected_val - actual_val)
                return max(0.0, 1.0 - (distance * 0.3))
        
        # For reasoning and speed, higher is generally better
        elif criterion in ["reasoning", "speed"]:
            if actual_val > expected_val:
                return 0.9  # Better than expected
            else:
                # Calculate distance penalty
                distance = abs(expected_val - actual_val)
                return max(0.0, 1.0 - (distance * 0.4))
        
        # Default distance-based scoring
        distance = abs(expected_val - actual_val)
        return max(0.0, 1.0 - (distance * 0.3))
    
    def _score_numeric_match(self, expected: str, actual: Any) -> float:
        """Score numeric criteria with range-based matching."""
        try:
            expected_num = float(expected)
            actual_num = float(actual)
            
            # Exact match
            if expected_num == actual_num:
                return 1.0
            
            # Calculate relative difference
            if expected_num == 0:
                return 0.0
            
            relative_diff = abs(expected_num - actual_num) / expected_num
            
            # Score based on relative difference
            if relative_diff <= 0.1:  # Within 10%
                return 0.9
            elif relative_diff <= 0.25:  # Within 25%
                return 0.7
            elif relative_diff <= 0.5:  # Within 50%
                return 0.5
            else:
                return 0.2
        
        except (ValueError, TypeError):
            return 0.0
    
    def _score_string_match(self, expected: str, actual: str) -> float:
        """Score string criteria with partial matching."""
        # Exact match
        if expected == actual:
            return 1.0
        
        # Substring match
        if expected in actual or actual in expected:
            return 0.7
        
        # Check for common words
        expected_words = set(expected.split())
        actual_words = set(actual.split())
        
        if expected_words & actual_words:  # Intersection
            overlap = len(expected_words & actual_words)
            total = len(expected_words | actual_words)
            return 0.3 + (0.4 * overlap / total)
        
        return 0.0
    
    def _is_numeric(self, value: Any) -> bool:
        """Check if a value can be converted to a number."""
        try:
            float(str(value))
            return True
        except (ValueError, TypeError):
            return False
    
    def _generate_suggestions(self, criteria: Dict[str, str], models: Dict[str, ModelConfig]) -> List[str]:
        """
        Generate helpful suggestions when no models match the criteria.
        
        Enhanced to support custom metadata fields defined by users and provide
        more actionable suggestions for improving model configurations.
        
        Args:
            criteria: The criteria that failed to match
            models: Available models
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        # Analyze available metadata and model fields to suggest alternatives
        available_metadata = {}
        available_providers = set()
        available_model_names = set()
        models_with_metadata = 0
        
        for model_config in models.values():
            # Collect providers and model names
            available_providers.add(model_config.provider)
            available_model_names.add(model_config.model)
            
            # Count models with metadata
            if model_config.metadata:
                models_with_metadata += 1
            
            # Collect metadata (including custom fields)
            for key, value in model_config.metadata.items():
                if key not in available_metadata:
                    available_metadata[key] = set()
                available_metadata[key].add(str(value).lower())
        
        # Check if models have sufficient metadata for intent resolution
        if models_with_metadata < len(models) * 0.5:
            suggestions.append(f"Only {models_with_metadata}/{len(models)} models have metadata - add metadata to more models for better intent resolution")
        
        # Suggest alternatives for each criterion
        for criterion, expected_value in criteria.items():
            if criterion == "provider":
                available_values = sorted(available_providers)
                suggestions.append(f"Available providers: {', '.join(available_values)}")
            elif criterion == "model":
                available_values = sorted(available_model_names)
                suggestions.append(f"Available model names: {', '.join(available_values)}")
            elif criterion in available_metadata:
                available_values = sorted(available_metadata[criterion])
                suggestions.append(f"Available {criterion} values: {', '.join(available_values)}")
            else:
                # Custom metadata field not found
                suggestions.append(f"No models have '{criterion}' metadata - add this field to model metadata for intent matching")
                
                # Suggest similar metadata fields if available
                similar_fields = [field for field in available_metadata.keys() 
                                if criterion.lower() in field.lower() or field.lower() in criterion.lower()]
                if similar_fields:
                    suggestions.append(f"Similar metadata fields available: {', '.join(similar_fields)}")
        
        # Suggest common metadata fields if missing
        common_fields = ["reasoning", "cost", "speed", "capabilities"]
        missing_common_fields = [field for field in common_fields if field not in available_metadata]
        if missing_common_fields:
            suggestions.append(f"Consider adding common metadata fields: {', '.join(missing_common_fields)}")
        
        # Suggest relaxing criteria
        if len(criteria) > 1:
            suggestions.append("Try using fewer criteria or different values")
        
        # Suggest specific models if available
        if models:
            model_names = list(models.keys())[:3]  # Show first 3 models
            suggestions.append(f"Available models: {', '.join(model_names)}")
        
        # Suggest using ConfigurationTemplates for better metadata
        suggestions.append("Use ConfigurationTemplates to generate models with comprehensive metadata")
        
        return suggestions
    
    def set_weights(self, weights: Dict[str, float]) -> None:
        """
        Set custom weights for criteria matching.
        
        Supports both standard and custom metadata fields defined by users.
        
        Args:
            weights: Dictionary mapping criterion names to weight values
        """
        self._default_weights.update(weights)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current criterion weights."""
        return self._default_weights.copy()
    
    def add_custom_ordinal_mapping(self, field: str, mapping: Dict[str, int]) -> None:
        """
        Add custom ordinal mapping for user-defined metadata fields.
        
        This allows users to define custom ordinal scales for their metadata fields,
        enabling proper scoring for custom criteria like "complexity", "accuracy", etc.
        
        Args:
            field: Name of the metadata field
            mapping: Dictionary mapping string values to numeric order
            
        Example:
            matcher.add_custom_ordinal_mapping("accuracy", {"low": 1, "medium": 2, "high": 3})
        """
        self._ordinal_mappings[field] = mapping
    
    def get_available_metadata_fields(self) -> Dict[str, set]:
        """
        Get all available metadata fields and their possible values from current configuration.
        
        This helps users understand what metadata fields are available for intent matching
        and what values they can use in their URN queries.
        
        Returns:
            Dictionary mapping field names to sets of possible values
        """
        models = self._config_manager.get_models()
        available_metadata = {}
        
        for model_config in models.values():
            # Add provider and model as special fields
            if "provider" not in available_metadata:
                available_metadata["provider"] = set()
            available_metadata["provider"].add(model_config.provider)
            
            if "model" not in available_metadata:
                available_metadata["model"] = set()
            available_metadata["model"].add(model_config.model)
            
            # Add all metadata fields
            for key, value in model_config.metadata.items():
                if key not in available_metadata:
                    available_metadata[key] = set()
                available_metadata[key].add(str(value).lower())
        
        return available_metadata



class ResolutionEngine:
    """
    Central engine for resolving model identifiers to concrete configurations.
    
    Handles resolution of physical identifiers, profile-based resolution,
    alias chains, and intent-based URN resolution.
    """
    
    def __init__(self, configuration_manager):
        """
        Initialize resolution engine with configuration manager.
        
        Args:
            configuration_manager: ConfigurationManager instance
        """
        self._config_manager = configuration_manager
        self._alias_resolver = AliasResolver(configuration_manager)
        self._intent_matcher = IntentMatcher(configuration_manager)
        
        # Physical identifier pattern: provider/model
        self._physical_pattern = re.compile(r'^([a-zA-Z0-9_-]+)/([a-zA-Z0-9_.-]+)$')
    
    @cache_resolution
    def resolve_identifier(self, identifier: Optional[str]) -> ResolvedModel:
        """
        Resolve any type of identifier to concrete model configuration.
        
        This is the main entry point for resolution. It determines the type
        of identifier and delegates to the appropriate resolution method.
        
        Args:
            identifier: Model identifier (URN, profile, alias, or physical ID).
                       If None, uses the default profile.
            
        Returns:
            ResolvedModel instance
            
        Raises:
            ResolutionError: If identifier cannot be resolved
        """
        try:
            # Handle None identifier - use default profile
            if identifier is None:
                default_profile = self._config_manager.get_default_profile()
                if default_profile:
                    return self.resolve_profile(default_profile)
                else:
                    raise ResolutionError(
                        "No identifier provided and no default profile configured",
                        identifier=identifier
                    )
            
            # Determine identifier type and resolve accordingly
            if self._is_urn(identifier):
                return self.resolve_intent(identifier)
            elif self._is_physical_identifier(identifier):
                return self.resolve_physical(identifier)
            elif self._config_manager.has_profile(identifier):
                return self.resolve_profile(identifier)
            elif self._config_manager.has_alias(identifier):
                return self.resolve_alias(identifier)
            elif self._config_manager.has_model(identifier):
                return self.resolve_model_id(identifier)
            else:
                # Enhanced error handling with configuration examples
                from .config_templates import ConfigurationExamples
                
                available_models = list(self._config_manager.get_models().keys())
                suggestions = ConfigurationExamples.get_provider_suggestions(available_models)
                configuration_examples = []
                
                # Try to provide a configuration example if it looks like a model identifier
                if not any(char in identifier for char in ['/', ':', '?']):
                    # Looks like a simple model identifier
                    example = ConfigurationExamples.get_model_example(identifier)
                    configuration_examples.append(example)
                
                raise ResolutionError(
                    f"Unknown identifier type or identifier not found: '{identifier}'",
                    identifier=identifier,
                    available_models=available_models,
                    suggestions=suggestions,
                    configuration_examples=configuration_examples,
                    quick_start_instructions=ConfigurationExamples.get_quick_start_instructions() if not available_models else None
                )
        
        except ResolutionError:
            # Re-raise resolution errors as-is
            raise
        except Exception as e:
            raise ResolutionError(
                f"Unexpected error during resolution: {str(e)}",
                identifier=identifier,
                context={"error_type": type(e).__name__}
            )
    
    def resolve_physical(self, physical_id: str) -> ResolvedModel:
        """
        Resolve direct provider/model reference.
        
        Physical identifiers have the format "provider/model" and directly
        specify both the provider and model name.
        
        Args:
            physical_id: Physical identifier (e.g., "openai/gpt-4")
            
        Returns:
            ResolvedModel instance
            
        Raises:
            ResolutionError: If physical ID cannot be resolved
        """
        if not self._is_physical_identifier(physical_id):
            raise ResolutionError(
                f"Invalid physical identifier format: '{physical_id}'. Expected 'provider/model'",
                identifier=physical_id
            )
        
        match = self._physical_pattern.match(physical_id)
        provider_name = match.group(1)
        model_name = match.group(2)
        
        # Check if provider exists
        provider_config = self._config_manager.get_provider(provider_name)
        if not provider_config:
            from .config_templates import ConfigurationExamples
            
            available_providers = list(self._config_manager.get_providers().keys())
            suggestions = [
                f"Available providers: {', '.join(available_providers)}" if available_providers else "No providers configured",
                f"Add provider '{provider_name}' to your configuration",
                "Use ConfigurationTemplates to generate provider configurations"
            ]
            
            # Provide configuration example for the missing provider
            configuration_examples = [ConfigurationExamples.get_model_example(model_name, provider_name)]
            
            raise ResolutionError(
                f"Unknown provider '{provider_name}' in physical identifier '{physical_id}'",
                identifier=physical_id,
                available_models=available_providers,
                suggestions=suggestions,
                configuration_examples=configuration_examples
            )
        
        # Create model configuration for physical identifier
        model_config = ModelConfig(
            provider=provider_name,
            model=model_name,
            metadata={},
            parameters={}
        )
        
        # Create resolved model
        return ResolvedModel(
            config=model_config,
            provider_config=provider_config,
            effective_parameters={}
        )
    
    def resolve_profile(self, profile_name: str) -> ResolvedModel:
        """
        Resolve profile-based reference using logical profile names.
        
        Profiles reference models and can include profile-specific parameters
        that override model defaults.
        
        Args:
            profile_name: Name of the profile to resolve
            
        Returns:
            ResolvedModel instance
            
        Raises:
            ResolutionError: If profile cannot be resolved
        """
        profile = self._config_manager.get_profile(profile_name)
        if not profile:
            available_profiles = list(self._config_manager.get_profiles().keys())
            available_models = list(self._config_manager.get_models().keys())
            
            suggestions = [
                f"Available profiles: {', '.join(available_profiles[:5])}" if available_profiles else "No profiles configured",
                "Create a profile that references an existing model",
                "Or use a model ID directly"
            ]
            
            if available_models:
                suggestions.append(f"Available models: {', '.join(available_models[:5])}")
            
            raise ResolutionError(
                f"Profile '{profile_name}' not found in configuration",
                identifier=profile_name,
                available_models=available_profiles,
                suggestions=suggestions
            )
        
        # Resolve the model reference in the profile
        try:
            resolved_model = self.resolve_identifier(profile.model)
        except ResolutionError as e:
            raise ResolutionError(
                f"Failed to resolve model '{profile.model}' referenced by profile '{profile_name}': {e.message}",
                identifier=profile_name,
                resolution_path=[profile_name, profile.model],
                context=e.context
            )
        
        # Merge profile parameters with resolved model parameters
        effective_parameters = resolved_model.effective_parameters.copy()
        effective_parameters.update(profile.parameters)
        
        # Return new resolved model with merged parameters
        return ResolvedModel(
            config=resolved_model.config,
            provider_config=resolved_model.provider_config,
            effective_parameters=effective_parameters
        )
    
    def resolve_model_id(self, model_id: str) -> ResolvedModel:
        """
        Resolve a direct model ID reference.
        
        Args:
            model_id: Model identifier from configuration
            
        Returns:
            ResolvedModel instance
            
        Raises:
            ResolutionError: If model ID cannot be resolved
        """
        model_config = self._config_manager.get_model(model_id)
        if not model_config:
            from .config_templates import ConfigurationExamples
            
            available_models = list(self._config_manager.get_models().keys())
            suggestions = ConfigurationExamples.get_provider_suggestions(available_models)
            configuration_examples = [ConfigurationExamples.get_model_example(model_id)]
            
            raise ResolutionError(
                f"Model '{model_id}' not found in configuration",
                identifier=model_id,
                available_models=available_models,
                suggestions=suggestions,
                configuration_examples=configuration_examples
            )
        
        # Get provider configuration
        provider_config = self._config_manager.get_provider(model_config.provider)
        if not provider_config:
            raise ResolutionError(
                f"Provider '{model_config.provider}' not found for model '{model_id}'",
                identifier=model_id,
                context={
                    "available_providers": list(self._config_manager.get_providers().keys())
                }
            )
        
        # Create resolved model with model's default parameters
        return ResolvedModel(
            config=model_config,
            provider_config=provider_config,
            effective_parameters=model_config.parameters.copy()
        )
    
    def resolve_intent(self, urn: str) -> ResolvedModel:
        """
        Resolve URN-based intent to best matching model using sophisticated scoring.
        
        Uses the IntentMatcher to find the best model based on metadata scoring,
        with support for exact matches, partial matches, and weighted criteria.
        
        Args:
            urn: URN string with intent parameters
            
        Returns:
            ResolvedModel instance
            
        Raises:
            ResolutionError: If no matching model found or URN is invalid
        """
        try:
            # Parse URN using the URNParser
            service, criteria = URNParser.parse(urn)
            
            # Use IntentMatcher to find the best matching model
            best_model_id = self._intent_matcher.find_best_match(service, criteria)
            
            # Resolve the best matching model to a concrete configuration
            return self.resolve_model_id(best_model_id)
            
        except ResolutionError:
            # Re-raise resolution errors as-is (includes URN parsing errors)
            raise
        except Exception as e:
            raise ResolutionError(
                f"Unexpected error during intent resolution: {str(e)}",
                identifier=urn,
                context={
                    "error_type": type(e).__name__,
                    "error_message": str(e)
                }
            )
    
    def resolve_alias(self, alias: str) -> ResolvedModel:
        """
        Resolve alias chain to concrete model configuration.
        
        Uses the AliasResolver to follow alias chains while detecting circular
        references to prevent infinite loops.
        
        Args:
            alias: Alias name
            
        Returns:
            ResolvedModel instance
            
        Raises:
            ResolutionError: If alias cannot be resolved
            CircularAliasError: If circular reference detected
        """
        try:
            # Use AliasResolver to get the final target
            final_target = self._alias_resolver.resolve_alias(alias)
            
            # Resolve the final target to a concrete model configuration
            return self.resolve_identifier(final_target)
            
        except (CircularAliasError, ResolutionError):
            # Re-raise alias resolution errors as-is
            raise
        except Exception as e:
            # Get resolution chain for context
            try:
                resolution_chain = self._alias_resolver.resolve_chain(alias)
            except:
                resolution_chain = [alias]
            
            raise ResolutionError(
                f"Failed to resolve alias '{alias}': {str(e)}",
                identifier=alias,
                resolution_path=resolution_chain,
                context={"error_type": type(e).__name__}
            )
    

    
    def get_alias_resolver(self) -> AliasResolver:
        """
        Get the alias resolver instance.
        
        Returns:
            AliasResolver instance
        """
        return self._alias_resolver
    
    def get_intent_matcher(self) -> IntentMatcher:
        """
        Get the intent matcher instance.
        
        Returns:
            IntentMatcher instance
        """
        return self._intent_matcher
    
    def resolve_alias_chain(self, alias: str) -> List[str]:
        """
        Show the full resolution path for an alias for debugging.
        
        This method delegates to the AliasResolver to provide the complete
        chain from the initial alias to the final target.
        
        Args:
            alias: Alias name to trace
            
        Returns:
            List of identifiers showing the full resolution path
            
        Raises:
            ResolutionError: If alias cannot be resolved
            CircularAliasError: If circular reference detected
        """
        return self._alias_resolver.resolve_chain(alias)
    
    def find_models_by_intent(self, urn: str, min_score: float = 0.1) -> List[Tuple[str, float]]:
        """
        Find all models matching an intent URN with their scores.
        
        Args:
            urn: URN string with intent parameters
            min_score: Minimum score threshold (0.0 to 1.0)
            
        Returns:
            List of (model_id, score) tuples sorted by score (highest first)
            
        Raises:
            ResolutionError: If URN format is invalid
        """
        try:
            service, criteria = URNParser.parse(urn)
            return self._intent_matcher.find_all_matches(service, criteria, min_score)
        except ResolutionError:
            raise
        except Exception as e:
            raise ResolutionError(
                f"Error finding models by intent: {str(e)}",
                identifier=urn,
                context={"error_type": type(e).__name__}
            )
    
    def parse_urn(self, urn: str) -> Tuple[str, Dict[str, str]]:
        """
        Parse a URN string into service and parameters.
        
        Args:
            urn: URN string to parse
            
        Returns:
            Tuple of (service, parameters_dict)
            
        Raises:
            ResolutionError: If URN format is invalid
        """
        return URNParser.parse(urn)
    
    def _is_physical_identifier(self, identifier: str) -> bool:
        """Check if identifier is a physical identifier (provider/model format)."""
        return bool(self._physical_pattern.match(identifier))
    
    def _is_urn(self, identifier: str) -> bool:
        """Check if identifier is a URN."""
        return URNParser.is_valid_urn(identifier)
    
    def _get_resolution_suggestions(self, identifier: str) -> List[str]:
        """
        Get context-aware suggestions for resolving an unknown identifier.
        
        Enhanced to provide more specific suggestions based on identifier patterns
        and available configuration options.
        
        Args:
            identifier: The identifier that failed to resolve
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Check if it looks like a physical identifier but has issues
        if '/' in identifier:
            provider_name = identifier.split('/')[0]
            available_providers = list(self._config_manager.get_providers().keys())
            
            if provider_name not in available_providers:
                suggestions.append(f"Provider '{provider_name}' not found. Available providers: {', '.join(available_providers)}")
                suggestions.append(f"Add provider '{provider_name}' to your configuration")
            else:
                suggestions.append("Check if the model name after '/' is correct")
                suggestions.append("Physical identifiers use format: provider/model")
        
        # Check if it looks like a URN but has issues
        elif identifier.startswith('urn:'):
            suggestions.append("Check URN format: urn:llm:intent:service?param=value")
            suggestions.append("Example: urn:llm:intent:chat?reasoning=high&cost=low")
            
            # Get available metadata fields for URN suggestions
            available_metadata = self._intent_matcher.get_available_metadata_fields()
            if available_metadata:
                common_fields = ["reasoning", "cost", "speed", "capabilities"]
                available_common = [f for f in common_fields if f in available_metadata]
                if available_common:
                    suggestions.append(f"Available metadata fields for URN: {', '.join(available_common)}")
        
        # Check if it looks like a simple model name
        elif not any(char in identifier for char in [':', '?', '/']):
            # Suggest similar model names
            models = list(self._config_manager.get_models().keys())
            similar_models = [m for m in models if identifier.lower() in m.lower() or m.lower() in identifier.lower()]
            if similar_models:
                suggestions.append(f"Similar models found: {', '.join(similar_models[:3])}")
            
            suggestions.append(f"Add model '{identifier}' to your configuration")
            suggestions.append("Use ConfigurationTemplates to generate model configurations")
        
        # Suggest available options by category
        profiles = list(self._config_manager.get_profiles().keys())
        if profiles:
            suggestions.append(f"Available profiles: {', '.join(profiles[:5])}")
        
        aliases = list(self._config_manager.get_aliases().keys())
        if aliases:
            suggestions.append(f"Available aliases: {', '.join(aliases[:5])}")
        
        models = list(self._config_manager.get_models().keys())
        if models:
            suggestions.append(f"Available models: {', '.join(models[:5])}")
            if len(models) > 5:
                suggestions.append(f"... and {len(models) - 5} more models")
        else:
            suggestions.append("No models configured - use ConfigurationTemplates to get started")
        
        return suggestions