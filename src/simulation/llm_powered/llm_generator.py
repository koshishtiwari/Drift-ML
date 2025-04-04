"""
LLM-powered data generator for the drift simulation system.
This module extends the BaseDataGenerator with LLM capabilities for more realistic data generation.
"""

import os
import json
import time
import requests
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import pandas as pd
import numpy as np
from .base_generator import BaseDataGenerator

logger = logging.getLogger(__name__)

class LLMDataGenerator(BaseDataGenerator):
    """LLM-powered data generator for more realistic synthetic data."""
    
    def __init__(self, config_path: str):
        """
        Initialize the LLM-powered data generator.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        super().__init__(config_path)
        
        # Initialize LLM client based on provider
        self.llm_config = self.config['llm']
        self.provider = self.llm_config.get('provider', 'openai')
        self.model_name = self.llm_config.get('model_name', 'gpt-3.5-turbo')
        
        # Cache for LLM generated content to reduce API calls
        self.text_cache = {}
        self.prompt_templates = self._initialize_prompt_templates()
        
        logger.info(f"Initialized LLMDataGenerator with provider: {self.provider}, model: {self.model_name}")
    
    def _initialize_prompt_templates(self) -> Dict[str, str]:
        """Initialize prompt templates for different LLM tasks."""
        return {
            'text_generation': """
            Generate {count} product reviews with the following characteristics:
            - Length: approximately {length} words
            - Sentiment: {sentiment} (0 = very negative, 1 = very positive)
            - Product category: {category}
            
            Return the reviews as a JSON array of strings. Each review should be realistic and varied.
            """,
            
            'sentiment_shift': """
            Rewrite the following product review to have a sentiment of {new_sentiment} (0 = very negative, 1 = very positive)
            while maintaining the same product details and overall structure.
            
            Original review: "{original_text}"
            
            Provide only the rewritten review text, without explanations or quotation marks.
            """,
            
            'concept_drift': """
            You are simulating a concept drift in a machine learning system.
            
            Original relationship: {original_relationship}
            
            Generate {count} examples where the relationship has changed to: {new_relationship}
            
            Return the examples as a JSON array of the form:
            [{"input_features": {...}, "output": value}, ...]
            
            Make sure the examples clearly demonstrate the new relationship pattern.
            """
        }
    
    def _call_llm_api(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Call the LLM API with the provided prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            temperature: Optional temperature parameter to override config
            
        Returns:
            The LLM's response text
        """
        if temperature is None:
            temperature = self.llm_config.get('temperature', 0.7)
            
        max_tokens = self.llm_config.get('max_tokens', 1000)
        
        if self.provider == 'openai':
            # OpenAI-compatible API call
            try:
                # Check for API key
                api_key = os.environ.get('OPENAI_API_KEY')
                if not api_key:
                    logger.warning("OPENAI_API_KEY not found in environment. Using mock responses.")
                    return self._generate_mock_response(prompt)
                
                # Real API call
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'model': self.model_name,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': temperature,
                    'max_tokens': max_tokens
                }
                
                response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers=headers,
                    json=data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    logger.error(f"API call failed with status {response.status_code}: {response.text}")
                    return self._generate_mock_response(prompt)
                    
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {str(e)}")
                return self._generate_mock_response(prompt)
        
        elif self.provider == 'huggingface':
            # Placeholder for HuggingFace API integration
            logger.warning("HuggingFace API not fully implemented yet. Using mock responses.")
            return self._generate_mock_response(prompt)
            
        elif self.provider == 'local':
            # Placeholder for local model inference
            local_model_path = self.llm_config.get('local_model_path')
            if not local_model_path:
                logger.warning("Local model path not specified. Using mock responses.")
                return self._generate_mock_response(prompt)
                
            logger.warning("Local model inference not fully implemented yet. Using mock responses.")
            return self._generate_mock_response(prompt)
            
        else:
            logger.error(f"Unsupported LLM provider: {self.provider}")
            return self._generate_mock_response(prompt)
    
    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate a mock response when the LLM API is unavailable.
        This provides fallback functionality for testing.
        
        Args:
            prompt: The original prompt
            
        Returns:
            A simulated LLM response
        """
        # Check what kind of content is requested based on the prompt
        if "product reviews" in prompt.lower():
            # Extract parameters from the prompt
            count = 1
            if "{count}" in prompt:
                try:
                    count = int(prompt.split("{count}")[1].split("}")[0].strip())
                except:
                    count = 3
                    
            sentiment = 0.5
            if "sentiment:" in prompt.lower():
                try:
                    sentiment_str = prompt.lower().split("sentiment:")[1].split("(")[0].strip()
                    if "{sentiment}" in sentiment_str:
                        sentiment = 0.5
                    else:
                        sentiment = float(sentiment_str)
                except:
                    sentiment = 0.5
            
            # Generate mock reviews
            reviews = []
            sentiments = ["negative", "somewhat negative", "neutral", "somewhat positive", "very positive"]
            sentiment_idx = min(int(sentiment * 5), 4)
            sentiment_word = sentiments[sentiment_idx]
            
            for i in range(count):
                if sentiment < 0.3:
                    reviews.append(f"I'm not happy with this product. The quality is poor and it didn't meet my expectations. Would not recommend.")
                elif sentiment < 0.7:
                    reviews.append(f"This product is okay. It does what it's supposed to do, but nothing special. Might buy again.")
                else:
                    reviews.append(f"Excellent product! I'm very satisfied with my purchase. The quality is great and it exceeded my expectations. Highly recommended!")
            
            return json.dumps(reviews)
            
        elif "rewrite the following product review" in prompt.lower():
            # Mock sentiment rewrite
            new_sentiment = 0.5
            if "{new_sentiment}" in prompt:
                try:
                    sentiment_part = prompt.split("{new_sentiment}")[1].split(")")[0].strip()
                    new_sentiment = float(sentiment_part)
                except:
                    new_sentiment = 0.5
            
            if new_sentiment < 0.3:
                return "I'm not satisfied with this product. It didn't work as expected and the quality is disappointing."
            elif new_sentiment < 0.7:
                return "This product is adequate. It does the job, but there are some minor issues to consider."
            else:
                return "I'm extremely pleased with this purchase! The product quality is excellent and it works perfectly."
                
        elif "concept drift" in prompt.lower():
            # Mock concept drift examples
            return json.dumps([
                {"input_features": {"x": 1, "y": 2}, "output": 5},
                {"input_features": {"x": 2, "y": 3}, "output": 8},
                {"input_features": {"x": 3, "y": 4}, "output": 11}
            ])
            
        else:
            # Generic mock response
            return "This is a mock response from the LLM. In production, this would be replaced with actual LLM-generated content."
    
    def _generate_text_feature(self, feature_config: Dict[str, Any], size: int) -> List[str]:
        """
        Override the base class method to generate more realistic text features using LLM.
        
        Args:
            feature_config: Configuration for the feature
            size: Number of samples to generate
            
        Returns:
            List of generated text samples
        """
        feature_name = feature_config['name']
        length_range = feature_config.get('length_range', [10, 100])
        sentiment_range = feature_config.get('sentiment_range', [0, 1])
        
        # Use cache to avoid regenerating the same content
        cache_key = f"{feature_name}_{size}"
        if cache_key in self.text_cache:
            return self.text_cache[cache_key][:size]  # Return cached results (up to size)
        
        # For efficiency, we'll generate batches of texts with the LLM instead of one at a time
        batch_size = min(size, 10)  # Generate up to 10 texts at a time
        texts = []
        
        for i in range(0, size, batch_size):
            current_batch_size = min(batch_size, size - i)
            
            # Randomize attributes for variety
            avg_length = self.rng.randint(length_range[0], length_range[1])
            avg_sentiment = self.rng.uniform(sentiment_range[0], sentiment_range[1])
            product_categories = ["electronics", "clothing", "home goods", "food", "toys"]
            category = self.rng.choice(product_categories)
            
            # Prepare prompt
            prompt = self.prompt_templates['text_generation'].format(
                count=current_batch_size,
                length=avg_length,
                sentiment=f"{avg_sentiment:.2f}",
                category=category
            )
            
            # Call LLM
            try:
                response = self._call_llm_api(prompt)
                batch_texts = json.loads(response)
                
                # Validate and process the response
                if isinstance(batch_texts, list):
                    texts.extend(batch_texts[:current_batch_size])
                else:
                    logger.warning(f"Invalid response format from LLM. Expected list, got {type(batch_texts)}")
                    # Fall back to base implementation
                    for _ in range(current_batch_size):
                        length = self.rng.randint(length_range[0], length_range[1])
                        sentiment = self.rng.uniform(sentiment_range[0], sentiment_range[1])
                        text = f"PLACEHOLDER_TEXT (length={length}, sentiment={sentiment:.2f})"
                        texts.append(text)
            except Exception as e:
                logger.error(f"Error generating text with LLM: {str(e)}")
                # Fall back to base implementation
                for _ in range(current_batch_size):
                    length = self.rng.randint(length_range[0], length_range[1])
                    sentiment = self.rng.uniform(sentiment_range[0], sentiment_range[1])
                    text = f"PLACEHOLDER_TEXT (length={length}, sentiment={sentiment:.2f})"
                    texts.append(text)
        
        # Cache the results
        self.text_cache[cache_key] = texts.copy()
        
        return texts
    
    def _apply_text_drift(self, texts: List[str], drift_params: Dict[str, Any]) -> List[str]:
        """
        Apply drift to text features using the LLM.
        
        Args:
            texts: Original text samples
            drift_params: Parameters defining the type of drift to apply
            
        Returns:
            List of drifted text samples
        """
        drifted_texts = []
        
        sentiment_shift = drift_params.get('sentiment_shift', 0.3)
        
        for text in texts:
            # Determine the new sentiment target
            # If current sentiment seems negative, make it more positive and vice versa
            if "not" in text.lower() or "poor" in text.lower() or "bad" in text.lower() or "disappointed" in text.lower():
                # Currently negative, shift to more positive
                new_sentiment = 0.7 + self.rng.uniform(0, 0.3)
            else:
                # Currently neutral or positive, shift to more negative
                new_sentiment = 0.3 - self.rng.uniform(0, 0.3)
            
            # Adjust by the sentiment_shift parameter
            new_sentiment = max(0, min(1, new_sentiment + sentiment_shift))
            
            # Prepare prompt
            prompt = self.prompt_templates['sentiment_shift'].format(
                new_sentiment=f"{new_sentiment:.2f}",
                original_text=text
            )
            
            # Call LLM
            try:
                drifted_text = self._call_llm_api(prompt)
                drifted_texts.append(drifted_text)
            except Exception as e:
                logger.error(f"Error applying text drift with LLM: {str(e)}")
                # Fall back to simple modification
                if new_sentiment > 0.5:
                    drifted_text = text.replace("not ", "").replace("poor", "good").replace("bad", "good")
                    drifted_text = "I really like this product! " + drifted_text
                else:
                    drifted_text = text.replace("great", "poor").replace("good", "bad")
                    drifted_text = "I'm not happy with this product. " + drifted_text
                drifted_texts.append(drifted_text)
        
        return drifted_texts
    
    def _apply_drift(self, features_df: pd.DataFrame, target: np.ndarray, 
                    drift_scenario: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Override the base class method to apply more sophisticated drift patterns using LLM.
        
        Args:
            features_df: DataFrame containing the generated features
            target: Array of target values
            drift_scenario: Dictionary defining the drift scenario
            
        Returns:
            Tuple of (drifted features DataFrame, drifted target array)
        """
        feature_name = drift_scenario.get('feature')
        drift_type = drift_scenario.get('type', 'feature_drift')
        drift_params = drift_scenario.get('drift_params', {})
        
        # Start with the base class implementation for standard drift types
        drifted_features, drifted_target = super()._apply_drift(features_df, target, drift_scenario)
        
        # Apply enhanced LLM-powered drift for text features
        if drift_type == 'feature_drift' and feature_name in drifted_features.columns:
            feature_config = next((f for f in self.features if f['name'] == feature_name), None)
            if not feature_config:
                return drifted_features, drifted_target
                
            feature_type = feature_config.get('type')
            
            if feature_type == 'text' and 'sentiment_shift' in drift_params:
                # Apply text drift using LLM
                logger.info(f"Applying LLM-powered text drift to {feature_name}")
                original_texts = drifted_features[feature_name].tolist()
                drifted_texts = self._apply_text_drift(original_texts, drift_params)
                drifted_features[feature_name] = drifted_texts
        
        # Apply enhanced concept drift that changes the relationship between features and target
        if drift_type == 'concept_drift':
            # This is a more complex case that would benefit from LLM guidance
            # In a real implementation, we would use the LLM to help define new relationships
            # For simplicity in this demonstration, we'll just modify the target based on the drift scenario
            
            if 'sentiment_shift' in drift_params and feature_name in drifted_features.columns:
                feature_config = next((f for f in self.features if f['name'] == feature_name), None)
                if feature_config and feature_config.get('type') == 'text':
                    # Modify the target based on text sentiment
                    logger.info(f"Applying concept drift to target based on {feature_name}")
                    
                    # Simple simulation: Adjust target values based on text content
                    sentiment_keywords = {
                        'positive': ['great', 'excellent', 'good', 'love', 'perfect', 'amazing'],
                        'negative': ['bad', 'poor', 'not', 'disappointed', 'terrible', 'worst']
                    }
                    
                    for i, text in enumerate(drifted_features[feature_name]):
                        text_lower = text.lower()
                        
                        # Count positive and negative keywords
                        pos_count = sum(1 for word in sentiment_keywords['positive'] if word in text_lower)
                        neg_count = sum(1 for word in sentiment_keywords['negative'] if word in text_lower)
                        
                        # Calculate sentiment score
                        if pos_count + neg_count > 0:
                            sentiment_score = pos_count / (pos_count + neg_count)
                        else:
                            sentiment_score = 0.5
                        
                        # Apply shift
                        target_range = self.target_config.get('range', [0, 1])
                        target_range_width = target_range[1] - target_range[0]
                        
                        # Adjust target based on sentiment and drift scenario
                        # Positive reviews lead to higher target values after drift
                        shift_amount = (sentiment_score - 0.5) * drift_params['sentiment_shift'] * target_range_width
                        drifted_target[i] += shift_amount
                    
                    # Clip to ensure values stay in range
                    drifted_target = np.clip(drifted_target, target_range[0], target_range[1])
        
        return drifted_features, drifted_target