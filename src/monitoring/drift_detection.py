"""
Model and data drift detection for Drift-ML platform.
"""
import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from loguru import logger

class DriftDetector:
    """Base class for drift detection."""
    
    def __init__(self, reference_data: Optional[pd.DataFrame] = None):
        """
        Initialize the drift detector.
        
        Args:
            reference_data: Reference data distribution
        """
        self.reference_data = reference_data
        self.is_fitted = False
    
    def fit(self, reference_data: pd.DataFrame) -> None:
        """
        Fit the detector to reference data.
        
        Args:
            reference_data: Reference data distribution
        """
        self.reference_data = reference_data
        self.is_fitted = True
    
    def detect(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current data distribution
            
        Returns:
            Dictionary with drift metrics
        """
        raise NotImplementedError("Subclasses must implement this method")

class StatisticalDriftDetector(DriftDetector):
    """
    Statistical tests for drift detection.
    Includes KS test, Chi-squared test, PSI, and JSI.
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        categorical_features: Optional[List[str]] = None,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.1,
        significance_level: float = 0.05
    ):
        """
        Initialize the statistical drift detector.
        
        Args:
            reference_data: Reference data distribution
            categorical_features: List of categorical feature names
            psi_threshold: Threshold for PSI to detect significant drift
            ks_threshold: Threshold for KS statistic to detect significant drift
            significance_level: Significance level for statistical tests
        """
        super().__init__(reference_data)
        self.categorical_features = categorical_features or []
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.significance_level = significance_level
        
        # Store distribution statistics for reference data
        self.reference_stats = {}
    
    def fit(self, reference_data: pd.DataFrame) -> None:
        """
        Fit the detector to reference data.
        
        Args:
            reference_data: Reference data distribution
        """
        super().fit(reference_data)
        
        # Calculate and store reference data statistics
        self.reference_stats = self._calculate_statistics(reference_data)
    
    def detect(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current data distribution
            
        Returns:
            Dictionary with drift metrics
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Calculate statistics for current data
        current_stats = self._calculate_statistics(current_data)
        
        # Compare distributions and calculate drift metrics
        drift_metrics = {}
        
        # Process numeric features
        numeric_features = [col for col in self.reference_data.columns 
                           if col not in self.categorical_features]
        
        # Calculate KS test for numeric features
        for feature in numeric_features:
            if feature in self.reference_data.columns and feature in current_data.columns:
                # KS test
                ks_stat, ks_pvalue = stats.ks_2samp(
                    self.reference_data[feature].dropna(),
                    current_data[feature].dropna()
                )
                
                drift_metrics[f"{feature}_ks_stat"] = ks_stat
                drift_metrics[f"{feature}_ks_pvalue"] = ks_pvalue
                drift_metrics[f"{feature}_ks_drift"] = ks_pvalue < self.significance_level
                
                # PSI
                psi_value = self._calculate_psi(
                    self.reference_stats.get(f"{feature}_bins", []),
                    current_stats.get(f"{feature}_bins", [])
                )
                
                drift_metrics[f"{feature}_psi"] = psi_value
                drift_metrics[f"{feature}_psi_drift"] = psi_value > self.psi_threshold
        
        # Process categorical features
        for feature in self.categorical_features:
            if feature in self.reference_data.columns and feature in current_data.columns:
                # Chi-squared test
                ref_counts = self.reference_stats.get(f"{feature}_counts", {})
                cur_counts = current_stats.get(f"{feature}_counts", {})
                
                # Get all categories
                all_categories = set(ref_counts.keys()) | set(cur_counts.keys())
                
                # Create observed and expected frequency arrays
                ref_total = sum(ref_counts.values())
                cur_total = sum(cur_counts.values())
                
                ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
                cur_freq = [cur_counts.get(cat, 0) for cat in all_categories]
                
                # Calculate chi-squared only if we have enough data
                if ref_total > 0 and cur_total > 0 and len(all_categories) > 1:
                    try:
                        chi2_stat, chi2_pvalue = stats.chisquare(cur_freq, f_exp=ref_freq)
                        drift_metrics[f"{feature}_chi2_stat"] = chi2_stat
                        drift_metrics[f"{feature}_chi2_pvalue"] = chi2_pvalue
                        drift_metrics[f"{feature}_chi2_drift"] = chi2_pvalue < self.significance_level
                    except Exception as e:
                        logger.warning(f"Chi-squared test failed for {feature}: {e}")
                
                # PSI for categorical features
                psi_value = self._calculate_categorical_psi(ref_counts, cur_counts)
                drift_metrics[f"{feature}_psi"] = psi_value
                drift_metrics[f"{feature}_psi_drift"] = psi_value > self.psi_threshold
        
        # Overall drift assessment
        feature_drift_flags = [v for k, v in drift_metrics.items() if k.endswith("_drift")]
        drift_metrics["any_drift_detected"] = any(feature_drift_flags)
        drift_metrics["drift_proportion"] = sum(feature_drift_flags) / len(feature_drift_flags) if feature_drift_flags else 0
        
        return drift_metrics
    
    def _calculate_statistics(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate statistics for data distribution.
        
        Args:
            data: Data to calculate statistics for
            
        Returns:
            Dictionary with distribution statistics
        """
        stats_dict = {}
        
        # Process numeric features
        numeric_features = [col for col in data.columns 
                           if col not in self.categorical_features]
        
        for feature in numeric_features:
            # Calculate basic statistics
            feature_data = data[feature].dropna()
            
            if len(feature_data) == 0:
                continue
            
            stats_dict[f"{feature}_mean"] = feature_data.mean()
            stats_dict[f"{feature}_median"] = feature_data.median()
            stats_dict[f"{feature}_std"] = feature_data.std()
            stats_dict[f"{feature}_min"] = feature_data.min()
            stats_dict[f"{feature}_max"] = feature_data.max()
            
            # Calculate histogram bins (for PSI)
            hist, bin_edges = np.histogram(feature_data, bins=10)
            total_count = len(feature_data)
            
            bin_data = []
            for i in range(len(hist)):
                bin_data.append({
                    "bin_start": bin_edges[i],
                    "bin_end": bin_edges[i+1],
                    "count": hist[i],
                    "proportion": hist[i] / total_count if total_count > 0 else 0
                })
            
            stats_dict[f"{feature}_bins"] = bin_data
        
        # Process categorical features
        for feature in self.categorical_features:
            if feature in data.columns:
                feature_data = data[feature].dropna()
                
                if len(feature_data) == 0:
                    continue
                
                # Calculate value counts
                value_counts = feature_data.value_counts().to_dict()
                stats_dict[f"{feature}_counts"] = value_counts
                
                # Calculate proportions
                total_count = len(feature_data)
                proportions = {k: v / total_count for k, v in value_counts.items()}
                stats_dict[f"{feature}_proportions"] = proportions
        
        return stats_dict
    
    def _calculate_psi(
        self,
        reference_bins: List[Dict[str, Any]],
        current_bins: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        Args:
            reference_bins: Histogram bins for reference data
            current_bins: Histogram bins for current data
            
        Returns:
            PSI value
        """
        # Handle empty inputs
        if not reference_bins or not current_bins:
            return 0.0
        
        # Ensure same number of bins and bin edges
        if len(reference_bins) != len(current_bins):
            return 0.0
        
        psi = 0.0
        
        for ref_bin, cur_bin in zip(reference_bins, current_bins):
            ref_prop = ref_bin["proportion"]
            cur_prop = cur_bin["proportion"]
            
            # Handle zero proportions (add small epsilon)
            ref_prop = max(ref_prop, 1e-6)
            cur_prop = max(cur_prop, 1e-6)
            
            # Calculate PSI component
            psi += (cur_prop - ref_prop) * np.log(cur_prop / ref_prop)
        
        return psi
    
    def _calculate_categorical_psi(
        self,
        reference_counts: Dict[str, int],
        current_counts: Dict[str, int]
    ) -> float:
        """
        Calculate PSI for categorical features.
        
        Args:
            reference_counts: Value counts for reference data
            current_counts: Value counts for current data
            
        Returns:
            PSI value
        """
        # Get total counts
        ref_total = sum(reference_counts.values())
        cur_total = sum(current_counts.values())
        
        # Handle empty inputs
        if ref_total == 0 or cur_total == 0:
            return 0.0
        
        # Get all categories
        all_categories = set(reference_counts.keys()) | set(current_counts.keys())
        
        psi = 0.0
        
        for category in all_categories:
            ref_count = reference_counts.get(category, 0)
            cur_count = current_counts.get(category, 0)
            
            ref_prop = ref_count / ref_total
            cur_prop = cur_count / cur_total
            
            # Handle zero proportions (add small epsilon)
            ref_prop = max(ref_prop, 1e-6)
            cur_prop = max(cur_prop, 1e-6)
            
            # Calculate PSI component
            psi += (cur_prop - ref_prop) * np.log(cur_prop / ref_prop)
        
        return psi

class OutlierDriftDetector(DriftDetector):
    """
    Outlier-based drift detection.
    Uses isolation forest or local outlier factor to detect outliers.
    """
    
    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        contamination: float = 0.05,
        method: str = "isolation_forest",
        features: Optional[List[str]] = None
    ):
        """
        Initialize the outlier drift detector.
        
        Args:
            reference_data: Reference data distribution
            contamination: Expected proportion of outliers
            method: Detection method (isolation_forest or lof)
            features: Features to use for drift detection
        """
        super().__init__(reference_data)
        self.contamination = contamination
        self.method = method
        self.features = features
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, reference_data: pd.DataFrame) -> None:
        """
        Fit the detector to reference data.
        
        Args:
            reference_data: Reference data distribution
        """
        super().fit(reference_data)
        
        # Select features if specified
        if self.features:
            X = reference_data[self.features].select_dtypes(include=np.number)
        else:
            X = reference_data.select_dtypes(include=np.number)
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and fit the model
        if self.method == "isolation_forest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42
            )
        elif self.method == "lof":
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.model.fit(X_scaled)
    
    def detect(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect drift using outlier detection.
        
        Args:
            current_data: Current data distribution
            
        Returns:
            Dictionary with drift metrics
        """
        if not self.is_fitted:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Select features if specified
        if self.features:
            X = current_data[self.features].select_dtypes(include=np.number)
        else:
            X = current_data.select_dtypes(include=np.number)
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Predict outliers
        if self.method == "isolation_forest":
            scores = self.model.score_samples(X_scaled)
            # Lower score = more anomalous
            is_outlier = scores < np.percentile(scores, self.contamination * 100)
        else:  # lof
            scores = self.model.score_samples(X_scaled)
            # Lower score = more anomalous
            is_outlier = scores < np.percentile(scores, self.contamination * 100)
        
        # Calculate drift metrics
        outlier_count = np.sum(is_outlier)
        outlier_proportion = outlier_count / len(current_data)
        
        drift_metrics = {
            "outlier_count": outlier_count,
            "outlier_proportion": outlier_proportion,
            "drift_detected": outlier_proportion > self.contamination * 1.5,  # Threshold at 1.5x expected
            "outlier_score_mean": np.mean(scores),
            "outlier_score_min": np.min(scores),
            "outlier_score_max": np.max(scores)
        }
        
        return drift_metrics

class ModelDriftMonitor:
    """
    Monitor model predictions and performance drift.
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str,
        reference_features: Optional[pd.DataFrame] = None,
        reference_predictions: Optional[np.ndarray] = None,
        feature_drift_detector: Optional[DriftDetector] = None,
        prediction_drift_detector: Optional[DriftDetector] = None
    ):
        """
        Initialize the model drift monitor.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            reference_features: Reference feature distribution
            reference_predictions: Reference prediction distribution
            feature_drift_detector: Detector for feature drift
            prediction_drift_detector: Detector for prediction drift
        """
        self.model_name = model_name
        self.model_version = model_version
        
        # Initialize drift detectors
        self.feature_drift_detector = feature_drift_detector or StatisticalDriftDetector()
        self.prediction_drift_detector = prediction_drift_detector or StatisticalDriftDetector()
        
        # Fit drift detectors if reference data is provided
        if reference_features is not None:
            self.feature_drift_detector.fit(reference_features)
        
        if reference_predictions is not None:
            # Convert to DataFrame if necessary
            if isinstance(reference_predictions, np.ndarray):
                ref_preds_df = pd.DataFrame({"prediction": reference_predictions})
            else:
                ref_preds_df = reference_predictions
            
            self.prediction_drift_detector.fit(ref_preds_df)
    
    def monitor_batch(
        self,
        features: pd.DataFrame,
        predictions: Union[np.ndarray, pd.DataFrame],
        callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> Dict[str, Any]:
        """
        Monitor a batch of predictions for drift.
        
        Args:
            features: Input features
            predictions: Model predictions
            callback: Optional callback function for reporting drift
            
        Returns:
            Dictionary with drift metrics
        """
        # Convert predictions to DataFrame if necessary
        if isinstance(predictions, np.ndarray):
            preds_df = pd.DataFrame({"prediction": predictions})
        else:
            preds_df = predictions
        
        drift_metrics = {}
        
        # Detect feature drift if detector is fitted
        if self.feature_drift_detector.is_fitted:
            feature_drift = self.feature_drift_detector.detect(features)
            drift_metrics["feature_drift"] = feature_drift
        
        # Detect prediction drift if detector is fitted
        if self.prediction_drift_detector.is_fitted:
            prediction_drift = self.prediction_drift_detector.detect(preds_df)
            drift_metrics["prediction_drift"] = prediction_drift
        
        # Overall drift assessment
        feature_drift_detected = drift_metrics.get("feature_drift", {}).get("any_drift_detected", False)
        prediction_drift_detected = drift_metrics.get("prediction_drift", {}).get("any_drift_detected", False)
        
        drift_metrics["overall_drift_detected"] = feature_drift_detected or prediction_drift_detected
        
        # Call the callback if provided
        if callback and drift_metrics["overall_drift_detected"]:
            callback(drift_metrics)
        
        return drift_metrics
    
    def set_reference_data(
        self,
        features: Optional[pd.DataFrame] = None,
        predictions: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> None:
        """
        Set or update reference data for drift detection.
        
        Args:
            features: Reference feature distribution
            predictions: Reference prediction distribution
        """
        if features is not None:
            self.feature_drift_detector.fit(features)
        
        if predictions is not None:
            # Convert to DataFrame if necessary
            if isinstance(predictions, np.ndarray):
                preds_df = pd.DataFrame({"prediction": predictions})
            else:
                preds_df = predictions
            
            self.prediction_drift_detector.fit(preds_df)

# Example usage
if __name__ == "__main__":
    # Create reference data
    np.random.seed(42)
    reference_features = pd.DataFrame({
        "feature1": np.random.normal(0, 1, 1000),
        "feature2": np.random.normal(5, 2, 1000),
        "category": np.random.choice(["A", "B", "C"], 1000)
    })
    
    reference_predictions = np.random.normal(0.5, 0.2, 1000)
    
    # Create monitor
    monitor = ModelDriftMonitor(
        model_name="example_model",
        model_version="1",
        reference_features=reference_features,
        reference_predictions=reference_predictions,
        feature_drift_detector=StatisticalDriftDetector(
            categorical_features=["category"]
        )
    )
    
    # Create current data with drift
    current_features = pd.DataFrame({
        "feature1": np.random.normal(0.5, 1, 100),  # Mean shifted from 0 to 0.5
        "feature2": np.random.normal(5, 2, 100),
        "category": np.random.choice(["A", "B", "C", "D"], 100)  # Added new category D
    })
    
    current_predictions = np.random.normal(0.7, 0.3, 100)  # Mean shifted from 0.5 to 0.7
    
    # Monitor batch
    drift_metrics = monitor.monitor_batch(
        features=current_features,
        predictions=current_predictions,
        callback=lambda metrics: print(f"Drift detected: {metrics['overall_drift_detected']}")
    )
    
    print(json.dumps(drift_metrics, indent=2))