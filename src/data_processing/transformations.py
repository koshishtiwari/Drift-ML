"""
Standard transformations for Drift-ML data processing.
"""
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np
from loguru import logger

def filter_invalid_data(record_json: str) -> bool:
    """
    Filter out invalid data records.
    
    Args:
        record_json: JSON string representing a record
        
    Returns:
        True if record is valid, False otherwise
    """
    try:
        record = json.loads(record_json)
        
        # Check if record has required fields
        if not isinstance(record, dict):
            return False
        
        # Check for required fields (customize based on your data)
        required_fields = ["id", "timestamp", "features"]
        if not all(field in record for field in required_fields):
            return False
        
        # Check if features is a dictionary
        if not isinstance(record["features"], dict):
            return False
        
        # Check if features are not all null/None
        features = record["features"]
        if not any(value is not None for value in features.values()):
            return False
        
        return True
    
    except Exception as e:
        logger.error(f"Error in filter_invalid_data: {e}")
        return False

def normalize_features(record_json: str) -> str:
    """
    Normalize numeric features to standard range.
    
    Args:
        record_json: JSON string representing a record
        
    Returns:
        JSON string with normalized features
    """
    try:
        record = json.loads(record_json)
        features = record.get("features", {})
        
        # Create a new features dictionary for normalized values
        normalized_features = {}
        
        for key, value in features.items():
            # Skip non-numeric values
            if not isinstance(value, (int, float)) or value is None:
                normalized_features[key] = value
                continue
            
            # Apply min-max normalization based on feature type
            # This is a simple example - in production, you would use
            # pre-computed normalization parameters for each feature
            if key.startswith("temperature"):
                # Example: normalize temperature to [0,1] assuming range [-50,50]
                normalized_features[key] = (value + 50) / 100
            elif key.startswith("percentage"):
                # Percentage is already in [0,1]
                normalized_features[key] = value
            elif key.startswith("count"):
                # Example: normalize log count
                if value > 0:
                    normalized_features[key] = np.log1p(value) / 10
                else:
                    normalized_features[key] = 0
            else:
                # Default normalization: simple capping to [-1,1]
                normalized_features[key] = max(min(value, 1), -1)
        
        # Replace original features with normalized ones
        record["features"] = normalized_features
        record["normalized"] = True
        
        return json.dumps(record)
    
    except Exception as e:
        logger.error(f"Error in normalize_features: {e}")
        return record_json

def enrich_with_metadata(record_json: str) -> str:
    """
    Enrich record with additional metadata.
    
    Args:
        record_json: JSON string representing a record
        
    Returns:
        JSON string with additional metadata
    """
    try:
        record = json.loads(record_json)
        
        # Add processing timestamp
        now = datetime.utcnow()
        record["processing_timestamp"] = now.isoformat() + "Z"
        
        # Calculate lag if original timestamp exists
        if "timestamp" in record:
            try:
                original_ts = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
                lag_seconds = (now - original_ts).total_seconds()
                record["processing_lag_seconds"] = lag_seconds
            except:
                pass
        
        # Add version metadata
        record["processor_version"] = "1.0.0"
        
        return json.dumps(record)
    
    except Exception as e:
        logger.error(f"Error in enrich_with_metadata: {e}")
        return record_json
