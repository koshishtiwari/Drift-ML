#!/usr/bin/env python3
"""
Base data generator for Drift-ML simulation system.
Provides foundation classes for generating synthetic data with controlled drift patterns.
"""

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import random
import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaseGenerator(ABC):
    """
    Abstract base class for data generators.
    
    This class defines the interface for all data generators in the Drift-ML simulation system.
    Specific generators should inherit from this class and implement the required methods.
    """
    
    def __init__(self, 
                 config: Dict,
                 seed: Optional[int] = None):
        """
        Initialize the base generator.
        
        Args:
            config: Configuration dictionary with parameters for data generation
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed if seed is not None else random.randint(1, 10000)
        self.rng = np.random.RandomState(self.seed)
        self.features = config.get('features', {})
        self.num_rows = config.get('num_rows', 1000)
        self.time_column = config.get('time_column', 'timestamp')
        self.label_column = config.get('label_column', 'target')
        
        logger.info(f"Initialized {self.__class__.__name__} with seed {self.seed}")
        
    def generate_base_data(self) -> pd.DataFrame:
        """
        Generate a base dataset without any drift.
        
        Returns:
            DataFrame with synthetic data
        """
        logger.info(f"Generating base dataset with {self.num_rows} rows")
        
        data = {}
        
        # Generate timestamp column if specified
        if self.time_column:
            start_date = datetime.datetime.now() - datetime.timedelta(days=30)
            data[self.time_column] = [
                start_date + datetime.timedelta(minutes=i*10) 
                for i in range(self.num_rows)
            ]
        
        # Generate feature columns based on feature specifications
        for feature_name, feature_config in self.features.items():
            feature_type = feature_config.get('type', 'numeric')
            
            if feature_type == 'numeric':
                data[feature_name] = self._generate_numeric_feature(feature_config)
            elif feature_type == 'categorical':
                data[feature_name] = self._generate_categorical_feature(feature_config)
            elif feature_type == 'text':
                data[feature_name] = self._generate_text_feature(feature_config)
            elif feature_type == 'datetime':
                data[feature_name] = self._generate_datetime_feature(feature_config)
            else:
                logger.warning(f"Unknown feature type '{feature_type}' for feature '{feature_name}'")
        
        # Generate target column if specified
        if self.label_column and self.label_column not in data:
            label_config = self.config.get('label', {})
            if label_config.get('type', 'classification') == 'classification':
                data[self.label_column] = self._generate_classification_label(data, label_config)
            else:
                data[self.label_column] = self._generate_regression_label(data, label_config)
        
        # Create and return DataFrame
        df = pd.DataFrame(data)
        return df
    
    def _generate_numeric_feature(self, feature_config: Dict) -> np.ndarray:
        """Generate a numeric feature column."""
        min_val = feature_config.get('min', 0)
        max_val = feature_config.get('max', 100)
        distribution = feature_config.get('distribution', 'uniform')
        
        if distribution == 'uniform':
            return self.rng.uniform(min_val, max_val, self.num_rows)
        elif distribution == 'normal':
            mean = feature_config.get('mean', (min_val + max_val) / 2)
            std = feature_config.get('std', (max_val - min_val) / 6)
            return self.rng.normal(mean, std, self.num_rows)
        else:
            logger.warning(f"Unknown distribution '{distribution}', using uniform")
            return self.rng.uniform(min_val, max_val, self.num_rows)
    
    def _generate_categorical_feature(self, feature_config: Dict) -> np.ndarray:
        """Generate a categorical feature column."""
        categories = feature_config.get('categories', ['A', 'B', 'C'])
        probabilities = feature_config.get('probabilities', None)
        
        if probabilities is None:
            # Equal probability for all categories
            return self.rng.choice(categories, self.num_rows)
        else:
            # Ensure probabilities sum to 1
            probabilities = np.array(probabilities)
            probabilities = probabilities / probabilities.sum()
            return self.rng.choice(categories, self.num_rows, p=probabilities)
    
    def _generate_text_feature(self, feature_config: Dict) -> List[str]:
        """Generate a text feature column with simple patterns."""
        patterns = feature_config.get('patterns', ["Text sample {i}", "Another text {i}"])
        return [random.choice(patterns).format(i=i) for i in range(self.num_rows)]
    
    def _generate_datetime_feature(self, feature_config: Dict) -> List[datetime.datetime]:
        """Generate a datetime feature column."""
        start_date = datetime.datetime.fromisoformat(
            feature_config.get('start_date', (datetime.datetime.now() - datetime.timedelta(days=365)).isoformat())
        )
        end_date = datetime.datetime.fromisoformat(
            feature_config.get('end_date', datetime.datetime.now().isoformat())
        )
        
        delta = end_date - start_date
        seconds = delta.total_seconds()
        
        return [
            start_date + datetime.timedelta(seconds=self.rng.uniform(0, seconds)) 
            for _ in range(self.num_rows)
        ]
    
    def _generate_classification_label(self, data: Dict, label_config: Dict) -> np.ndarray:
        """Generate a classification target using a simple decision rule."""
        features = label_config.get('features', list(self.features.keys())[:2])
        num_classes = label_config.get('num_classes', 2)
        
        if not features:
            # Random labels if no features specified
            return self.rng.randint(0, num_classes, self.num_rows)
        
        # Simple decision rule based on feature thresholds
        probs = np.zeros((self.num_rows, num_classes))
        for i in range(num_classes):
            # Each class gets influenced by different features
            feature_subset = features[i % len(features):]
            for feature in feature_subset:
                if feature in data:
                    # For numeric features, use the value directly
                    if np.issubdtype(np.array(data[feature]).dtype, np.number):
                        probs[:, i] += data[feature] / np.max(data[feature])
                    # For categorical features, use one-hot encoding
                    else:
                        unique_values = np.unique(data[feature])
                        for j, val in enumerate(unique_values):
                            mask = np.array(data[feature]) == val
                            probs[mask, i] += j / len(unique_values)
        
        # Normalize probabilities
        row_sums = probs.sum(axis=1)
        probs = probs / row_sums[:, np.newaxis]
        
        # Sample from multinomial distribution
        labels = np.array([self.rng.choice(num_classes, p=p) for p in probs])
        
        return labels
    
    def _generate_regression_label(self, data: Dict, label_config: Dict) -> np.ndarray:
        """Generate a regression target using a simple linear combination."""
        features = label_config.get('features', list(self.features.keys())[:3])
        noise_level = label_config.get('noise', 0.1)
        
        if not features or not all(f in data for f in features):
            # Random values if no valid features specified
            return self.rng.uniform(0, 100, self.num_rows)
        
        # Initialize target with zeros
        target = np.zeros(self.num_rows)
        
        # Add contribution from each feature
        for feature in features:
            if feature in data:
                # For numeric features, use the value directly
                if np.issubdtype(np.array(data[feature]).dtype, np.number):
                    # Random weight for this feature
                    weight = self.rng.uniform(-1, 1)
                    target += weight * np.array(data[feature])
                # For categorical features, use one-hot encoding
                else:
                    unique_values = np.unique(data[feature])
                    for val in unique_values:
                        mask = np.array(data[feature]) == val
                        # Random weight for this category
                        weight = self.rng.uniform(-1, 1)
                        target[mask] += weight
        
        # Add noise
        noise = self.rng.normal(0, noise_level * np.std(target), self.num_rows)
        target += noise
        
        return target
    
    @abstractmethod
    def apply_drift(self, data: pd.DataFrame, drift_config: Dict) -> pd.DataFrame:
        """
        Apply drift to the data according to the specified configuration.
        
        Args:
            data: Input DataFrame with no drift
            drift_config: Configuration specifying the type and parameters of drift to apply
            
        Returns:
            DataFrame with drift applied
        """
        pass
    
    def generate_data_with_drift(self, drift_config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate synthetic data with specified drift patterns.
        
        Args:
            drift_config: Configuration for drift patterns to apply
            
        Returns:
            DataFrame with synthetic data and applied drift
        """
        # Generate base data first
        data = self.generate_base_data()
        
        # Apply drift if configuration is provided
        if drift_config:
            logger.info(f"Applying drift with config: {drift_config}")
            data = self.apply_drift(data, drift_config)
        
        return data
    
    def simulate_stream(self, 
                        drift_configs: List[Dict], 
                        segment_sizes: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Simulate a data stream with different drift patterns in different segments.
        
        Args:
            drift_configs: List of drift configurations to apply to different segments
            segment_sizes: List of segment sizes (rows). If None, segments will be of equal size.
            
        Returns:
            DataFrame representing the entire stream with various drift patterns
        """
        total_rows = self.num_rows
        
        if segment_sizes is None:
            # Equal-sized segments
            segment_size = total_rows // len(drift_configs)
            segment_sizes = [segment_size] * (len(drift_configs) - 1)
            # Last segment gets the remainder
            segment_sizes.append(total_rows - sum(segment_sizes))
        
        # Ensure we have a segment size for each drift config
        assert len(segment_sizes) == len(drift_configs), \
            "Number of segment sizes must match number of drift configurations"
        
        all_segments = []
        
        # Generate each segment with its drift pattern
        for i, (size, drift_config) in enumerate(zip(segment_sizes, drift_configs)):
            logger.info(f"Generating segment {i+1}/{len(drift_configs)} with {size} rows")
            
            # Temporarily override number of rows
            self.num_rows = size
            
            # Generate this segment
            segment_data = self.generate_data_with_drift(drift_config)
            
            # Add segment index column
            segment_data['segment'] = i
            
            all_segments.append(segment_data)
        
        # Restore original number of rows
        self.num_rows = total_rows
        
        # Combine all segments
        combined_data = pd.concat(all_segments, ignore_index=True)
        
        # If time column exists, ensure it's properly sequenced across segments
        if self.time_column in combined_data.columns:
            # Sort by original index to maintain proper time sequence
            combined_data = combined_data.sort_index()
            
            # Generate new timestamps that are properly sequenced
            start_date = datetime.datetime.now() - datetime.timedelta(days=30)
            new_timestamps = [
                start_date + datetime.timedelta(minutes=i*10) 
                for i in range(len(combined_data))
            ]
            combined_data[self.time_column] = new_timestamps
        
        return combined_data
    
    def save_data(self, data: pd.DataFrame, output_path: str):
        """
        Save generated data to a file.
        
        Args:
            data: DataFrame to save
            output_path: Path to save the data to
        """
        # Determine format from file extension
        format_ext = output_path.split('.')[-1].lower()
        
        if format_ext == 'csv':
            data.to_csv(output_path, index=False)
        elif format_ext in ['parquet', 'pq']:
            data.to_parquet(output_path, index=False)
        elif format_ext == 'json':
            data.to_json(output_path, orient='records', lines=True)
        else:
            logger.warning(f"Unknown format '{format_ext}', saving as CSV")
            data.to_csv(output_path, index=False)
        
        logger.info(f"Saved data to {output_path}")


class SimpleDriftGenerator(BaseGenerator):
    """
    A simple implementation of the BaseGenerator that can apply basic drift patterns.
    """
    
    def apply_drift(self, data: pd.DataFrame, drift_config: Dict) -> pd.DataFrame:
        """
        Apply drift to the dataset.
        
        Supports multiple drift types:
        - feature_drift: Changes distribution of specific features
        - label_drift: Changes relationship between features and labels
        - concept_drift: Changes the underlying data generation process
        - covariate_drift: Changes the input distribution without changing the relationship
        
        Args:
            data: Original DataFrame
            drift_config: Configuration specifying drift parameters
            
        Returns:
            DataFrame with drift applied
        """
        drifted_data = data.copy()
        drift_type = drift_config.get('type', 'feature_drift')
        
        if drift_type == 'feature_drift':
            drifted_data = self._apply_feature_drift(drifted_data, drift_config)
        elif drift_type == 'label_drift':
            drifted_data = self._apply_label_drift(drifted_data, drift_config)
        elif drift_type == 'concept_drift':
            drifted_data = self._apply_concept_drift(drifted_data, drift_config)
        elif drift_type == 'covariate_drift':
            drifted_data = self._apply_covariate_drift(drifted_data, drift_config)
        else:
            logger.warning(f"Unknown drift type '{drift_type}', no drift applied")
        
        return drifted_data
    
    def _apply_feature_drift(self, data: pd.DataFrame, drift_config: Dict) -> pd.DataFrame:
        """Apply drift to specific features."""
        feature_drifts = drift_config.get('features', {})
        
        for feature_name, feature_drift in feature_drifts.items():
            if feature_name not in data.columns:
                logger.warning(f"Feature '{feature_name}' not found in data, skipping")
                continue
            
            drift_magnitude = feature_drift.get('magnitude', 0.5)
            drift_type = feature_drift.get('drift_type', 'shift')
            
            if np.issubdtype(data[feature_name].dtype, np.number):
                # For numeric features
                if drift_type == 'shift':
                    # Add a constant shift
                    shift_value = drift_magnitude * data[feature_name].std()
                    data[feature_name] += shift_value
                elif drift_type == 'scale':
                    # Scale the values
                    data[feature_name] *= (1 + drift_magnitude)
                elif drift_type == 'noise':
                    # Add random noise
                    noise = self.rng.normal(0, drift_magnitude * data[feature_name].std(), len(data))
                    data[feature_name] += noise
                elif drift_type == 'distribution_change':
                    # Change the distribution (e.g., from normal to exponential)
                    mean = data[feature_name].mean()
                    std = data[feature_name].std()
                    if 'new_distribution' in feature_drift:
                        new_dist = feature_drift['new_distribution']
                        if new_dist == 'exponential':
                            data[feature_name] = self.rng.exponential(scale=mean, size=len(data))
                        elif new_dist == 'lognormal':
                            data[feature_name] = self.rng.lognormal(mean=np.log(mean), 
                                                                  sigma=std/mean, 
                                                                  size=len(data))
            else:
                # For categorical features
                if drift_type == 'probability_change':
                    # Change the distribution of categories
                    categories = data[feature_name].unique()
                    new_probs = feature_drift.get('new_probabilities', None)
                    
                    if new_probs is None:
                        # Generate random new probabilities
                        new_probs = self.rng.dirichlet(np.ones(len(categories)) * (1 - drift_magnitude))
                    
                    # Sample new values
                    data[feature_name] = self.rng.choice(
                        categories, 
                        size=len(data), 
                        p=new_probs
                    )
                elif drift_type == 'category_swap':
                    # Swap some values between categories
                    swap_prob = feature_drift.get('swap_probability', drift_magnitude)
                    categories = list(data[feature_name].unique())
                    
                    if len(categories) < 2:
                        continue
                    
                    swap_mask = self.rng.random(len(data)) < swap_prob
                    
                    for idx in np.where(swap_mask)[0]:
                        current_val = data.loc[idx, feature_name]
                        other_categories = [c for c in categories if c != current_val]
                        new_val = self.rng.choice(other_categories)
                        data.loc[idx, feature_name] = new_val
        
        return data
    
    def _apply_label_drift(self, data: pd.DataFrame, drift_config: Dict) -> pd.DataFrame:
        """Apply drift to the relationship between features and labels."""
        if self.label_column not in data.columns:
            logger.warning(f"Label column '{self.label_column}' not found, skipping label drift")
            return data
        
        drift_magnitude = drift_config.get('magnitude', 0.3)
        drift_type = drift_config.get('drift_type', 'random_flip')
        
        # Determine if we're working with classification or regression
        is_classification = np.issubdtype(data[self.label_column].dtype, np.integer)
        
        if is_classification:
            if drift_type == 'random_flip':
                # Randomly flip some labels
                flip_mask = self.rng.random(len(data)) < drift_magnitude
                
                if flip_mask.sum() > 0:
                    unique_labels = data[self.label_column].unique()
                    
                    for idx in np.where(flip_mask)[0]:
                        current_label = data.loc[idx, self.label_column]
                        other_labels = [l for l in unique_labels if l != current_label]
                        
                        if other_labels:
                            data.loc[idx, self.label_column] = self.rng.choice(other_labels)
            
            elif drift_type == 'class_imbalance':
                # Change class distribution
                unique_labels = data[self.label_column].unique()
                target_label = drift_config.get('target_class', int(unique_labels[0]))
                increase_prob = drift_config.get('increase', True)
                
                # Select indices to potentially change
                change_mask = self.rng.random(len(data)) < drift_magnitude
                
                if increase_prob:
                    # Increase frequency of target class
                    data.loc[change_mask, self.label_column] = target_label
                else:
                    # Decrease frequency of target class
                    other_labels = [l for l in unique_labels if l != target_label]
                    if not other_labels:
                        return data
                        
                    target_mask = (data[self.label_column] == target_label) & change_mask
                    data.loc[target_mask, self.label_column] = self.rng.choice(
                        other_labels, size=target_mask.sum())
        else:
            # Regression
            if drift_type == 'noise_increase':
                # Add noise to the labels
                noise = self.rng.normal(0, drift_magnitude * data[self.label_column].std(), len(data))
                data[self.label_column] += noise
                
            elif drift_type == 'trend':
                # Add a trend component
                trend_type = drift_config.get('trend_type', 'linear')
                
                if trend_type == 'linear':
                    trend = np.linspace(0, drift_magnitude * data[self.label_column].std(), len(data))
                    data[self.label_column] += trend
                    
                elif trend_type == 'seasonal':
                    frequency = drift_config.get('frequency', 10)
                    amplitude = drift_magnitude * data[self.label_column].std()
                    seasonal = amplitude * np.sin(np.linspace(0, frequency * 2 * np.pi, len(data)))
                    data[self.label_column] += seasonal
        
        return data
    
    def _apply_concept_drift(self, data: pd.DataFrame, drift_config: Dict) -> pd.DataFrame:
        """Apply concept drift by changing the relationship between features and label."""
        if self.label_column not in data.columns:
            logger.warning(f"Label column '{self.label_column}' not found, skipping concept drift")
            return data
        
        is_classification = np.issubdtype(data[self.label_column].dtype, np.integer)
        drift_magnitude = drift_config.get('magnitude', 0.5)
        drift_features = drift_config.get('features', list(self.features.keys())[:2])
        
        # Get features that exist in the data
        valid_features = [f for f in drift_features if f in data.columns]
        
        if not valid_features:
            logger.warning("No valid features found for concept drift")
            return data
        
        # Create a new relationship between features and label
        if is_classification:
            # Classification
            unique_labels = data[self.label_column].unique()
            num_classes = len(unique_labels)
            
            # Only modify a portion of the data
            drift_mask = self.rng.random(len(data)) < drift_magnitude
            
            if drift_mask.sum() > 0:
                # Create new labels based on different feature relationship
                new_labels = np.zeros(drift_mask.sum(), dtype=data[self.label_column].dtype)
                
                # Extract relevant features for the subset
                subset_data = data.loc[drift_mask, valid_features].copy()
                
                # Normalize numeric features
                for feature in valid_features:
                    if np.issubdtype(subset_data[feature].dtype, np.number):
                        subset_data[feature] = (subset_data[feature] - subset_data[feature].mean()) / subset_data[feature].std()
                
                # Create a new decision boundary
                for i, row in subset_data.iterrows():
                    feature_sum = sum(row[f] if np.issubdtype(row[f].dtype, np.number) else hash(row[f]) % 10 
                                      for f in valid_features)
                    new_labels[i - drift_mask[:i].sum()] = unique_labels[int(feature_sum) % num_classes]
                
                # Apply the new labels
                data.loc[drift_mask, self.label_column] = new_labels
        else:
            # Regression
            drift_mask = self.rng.random(len(data)) < drift_magnitude
            
            if drift_mask.sum() > 0:
                # Create new target values based on different relationship
                new_values = np.zeros(drift_mask.sum())
                
                # Extract relevant features for the subset
                subset_data = data.loc[drift_mask, valid_features].copy()
                
                # Normalize numeric features
                for feature in valid_features:
                    if np.issubdtype(subset_data[feature].dtype, np.number):
                        subset_data[feature] = (subset_data[feature] - subset_data[feature].mean()) / subset_data[feature].std()
                
                # Create new regression formula with random weights
                weights = {f: self.rng.uniform(-1, 1) for f in valid_features}
                
                for i, row in subset_data.iterrows():
                    feature_contribution = sum(
                        weights[f] * row[f] if np.issubdtype(row[f].dtype, np.number) else weights[f] * (hash(row[f]) % 10)
                        for f in valid_features
                    )
                    # Add some noise
                    noise = self.rng.normal(0, 0.1 * abs(feature_contribution))
                    new_values[i - drift_mask[:i].sum()] = feature_contribution + noise
                
                # Scale to match the original range
                original_std = data.loc[~drift_mask, self.label_column].std()
                original_mean = data.loc[~drift_mask, self.label_column].mean()
                new_values = (new_values - new_values.mean()) / new_values.std() * original_std + original_mean
                
                # Apply the new values
                data.loc[drift_mask, self.label_column] = new_values
        
        return data
    
    def _apply_covariate_drift(self, data: pd.DataFrame, drift_config: Dict) -> pd.DataFrame:
        """Apply covariate drift by changing the distribution of inputs without changing relationship."""
        drift_magnitude = drift_config.get('magnitude', 0.5)
        drift_features = drift_config.get('features', list(self.features.keys()))
        
        # Get features that exist in the data
        valid_features = [f for f in drift_features if f in data.columns]
        
        if not valid_features:
            logger.warning("No valid features found for covariate drift")
            return data
        
        # Apply drift to each feature
        for feature in valid_features:
            # Only apply to a subset of the data to create a gradual drift
            drift_mask = self.rng.random(len(data)) < drift_magnitude
            
            if drift_mask.sum() == 0:
                continue
                
            if np.issubdtype(data[feature].dtype, np.number):
                # For numeric features
                # Calculate drift parameters
                orig_mean = data[feature].mean()
                orig_std = data[feature].std()
                
                # Change mean and variance
                shift = drift_config.get('shift', 0.5) * orig_std
                scale = drift_config.get('scale', 1.5)
                
                # Apply transformation
                data.loc[drift_mask, feature] = (data.loc[drift_mask, feature] - orig_mean) * scale + orig_mean + shift
            else:
                # For categorical features
                categories = data[feature].unique()
                
                if len(categories) < 2:
                    continue
                
                # Calculate original distribution
                original_counts = data[feature].value_counts(normalize=True)
                
                # Generate new distribution
                alpha = np.ones(len(categories)) * (1 - drift_magnitude)
                for i, cat in enumerate(categories):
                    if cat in original_counts:
                        alpha[i] += original_counts[cat]
                
                new_probs = self.rng.dirichlet(alpha)
                
                # Sample new values
                new_values = self.rng.choice(
                    categories, 
                    size=drift_mask.sum(), 
                    p=new_probs
                )
                
                data.loc[drift_mask, feature] = new_values
        
        return data