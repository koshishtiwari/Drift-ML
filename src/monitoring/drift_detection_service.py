"""
Drift Detection Service for the Drift-ML platform.
This service monitors models and data for drift and triggers alerts.
"""
import os
import time
import json
import threading
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from loguru import logger
import redis
import psycopg2
from prometheus_client import start_http_server, Gauge, Counter, Summary
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends

from src.monitoring.drift_detection import StatisticalDriftDetector, OutlierDriftDetector
from src.monitoring.metrics import ModelMetrics
from src.security.security import Security

app = FastAPI(title="Drift-ML Drift Detection Service")

# Prometheus metrics
DRIFT_SCORE = Gauge('drift_score', 'Data drift score by feature', ['model_name', 'model_version', 'feature'])
DRIFT_ALERT_COUNT = Counter('drift_alerts_total', 'Number of drift alerts triggered', ['model_name', 'model_version', 'severity'])
PROCESSING_TIME = Summary('drift_detection_processing_seconds', 'Time spent processing drift detection')

class DriftDetectionService:
    """Service for detecting data and model drift."""
    
    def __init__(
        self,
        reference_data_store_uri: str,
        production_data_store_uri: str,
        notification_uri: Optional[str] = None,
        window_size: int = 1000,
        drift_threshold: float = 0.1,
        security: Optional[Security] = None
    ):
        """
        Initialize the drift detection service.
        
        Args:
            reference_data_store_uri: URI for reference data storage
            production_data_store_uri: URI for production data storage
            notification_uri: URI for sending notifications
            window_size: Size of data window to analyze
            drift_threshold: Threshold for drift detection
            security: Security module instance
        """
        self.reference_data_store_uri = reference_data_store_uri
        self.production_data_store_uri = production_data_store_uri
        self.notification_uri = notification_uri
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.security = security
        
        # Initialize detectors
        self.statistical_detector = StatisticalDriftDetector(drift_threshold=drift_threshold)
        self.outlier_detector = OutlierDriftDetector(contamination=0.05)
        
        # Cache of model metrics instances
        self.metrics_cache: Dict[str, ModelMetrics] = {}
        
        # Background monitoring flag
        self.monitoring_active = False
    
    def get_model_metrics(self, model_name: str, model_version: str) -> ModelMetrics:
        """Get or create metrics for a model."""
        key = f"{model_name}/{model_version}"
        if key not in self.metrics_cache:
            self.metrics_cache[key] = ModelMetrics(model_name, model_version)
        return self.metrics_cache[key]
    
    def get_reference_data(self, model_name: str, model_version: str) -> pd.DataFrame:
        """
        Get reference data for a model.
        
        Returns:
            DataFrame containing reference data
        """
        # Implementation depends on data store type
        if self.reference_data_store_uri.startswith('postgres'):
            # Fetch from PostgreSQL
            try:
                conn = psycopg2.connect(self.reference_data_store_uri)
                query = f"""
                SELECT * FROM reference_data 
                WHERE model_name = '{model_name}' AND model_version = '{model_version}'
                LIMIT {self.window_size}
                """
                return pd.read_sql(query, conn)
            except Exception as e:
                logger.error(f"Failed to fetch reference data: {e}")
                return pd.DataFrame()
        else:
            # Default: read from local file
            try:
                file_path = f"data/reference/{model_name}/{model_version}/reference_data.parquet"
                if os.path.exists(file_path):
                    return pd.read_parquet(file_path)
                else:
                    logger.error(f"Reference data file not found: {file_path}")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Failed to load reference data: {e}")
                return pd.DataFrame()
    
    def get_production_data(self, model_name: str, model_version: str) -> pd.DataFrame:
        """
        Get recent production data for a model.
        
        Returns:
            DataFrame containing production data
        """
        # Implementation depends on data store type  
        if self.production_data_store_uri.startswith('redis'):
            # Fetch from Redis
            try:
                r = redis.Redis.from_url(self.production_data_store_uri)
                key = f"drift:input:{model_name}:{model_version}"
                data = r.lrange(key, 0, self.window_size - 1)
                records = [json.loads(item) for item in data]
                return pd.DataFrame(records)
            except Exception as e:
                logger.error(f"Failed to fetch production data from Redis: {e}")
                return pd.DataFrame()
        else:
            # Default: read from local file  
            try:
                file_path = f"data/production/{model_name}/{model_version}/recent_data.parquet"
                if os.path.exists(file_path):
                    return pd.read_parquet(file_path)
                else:
                    logger.error(f"Production data file not found: {file_path}")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Failed to load production data: {e}")
                return pd.DataFrame()
    
    @PROCESSING_TIME.time()
    def detect_drift(self, model_name: str, model_version: str) -> Dict[str, Any]:
        """
        Detect drift for a specific model version.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            
        Returns:
            Dictionary with drift detection results
        """
        reference_data = self.get_reference_data(model_name, model_version)
        production_data = self.get_production_data(model_name, model_version)
        
        if reference_data.empty or production_data.empty:
            logger.warning(f"Empty reference or production data for {model_name}/{model_version}")
            return {
                "model_name": model_name,
                "model_version": model_version,
                "status": "error",
                "error": "Empty reference or production data",
                "drift_detected": False
            }
        
        # Ensure column compatibility
        common_columns = list(set(reference_data.columns) & set(production_data.columns))
        if not common_columns:
            logger.error(f"No common columns between reference and production data for {model_name}/{model_version}")
            return {
                "model_name": model_name,
                "model_version": model_version,
                "status": "error",
                "error": "No common columns between reference and production data",
                "drift_detected": False
            }
        
        reference_data = reference_data[common_columns]
        production_data = production_data[common_columns]
        
        # Detect drift using statistical methods
        drift_scores = {}
        drift_detected = False
        
        # Calculate per-feature drift
        numeric_columns = reference_data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = list(set(common_columns) - set(numeric_columns))
        
        for feature in numeric_columns:
            ref_values = reference_data[feature].dropna().values
            prod_values = production_data[feature].dropna().values
            
            if len(ref_values) == 0 or len(prod_values) == 0:
                continue
                
            drift_score = self.statistical_detector.detect_drift_feature(
                reference_values=ref_values,
                current_values=prod_values
            )
            
            drift_scores[feature] = drift_score
            DRIFT_SCORE.labels(
                model_name=model_name,
                model_version=model_version,
                feature=feature
            ).set(drift_score)
            
            # Record drift score in model metrics
            metrics = self.get_model_metrics(model_name, model_version)
            metrics.record_drift_score(feature, drift_score)
            
            if drift_score > self.drift_threshold:
                drift_detected = True
        
        # Global drift detection using outlier method on numeric features
        if numeric_columns:
            try:
                global_drift = self.outlier_detector.detect_drift(
                    reference_data=reference_data[numeric_columns],
                    current_data=production_data[numeric_columns]
                )
                
                if global_drift > self.drift_threshold:
                    drift_detected = True
                    DRIFT_ALERT_COUNT.labels(
                        model_name=model_name, 
                        model_version=model_version,
                        severity="warning"
                    ).inc()
                    
                    # Send alert if notification URI is configured
                    if self.notification_uri:
                        self._send_drift_alert(
                            model_name=model_name,
                            model_version=model_version,
                            drift_score=global_drift,
                            details=drift_scores
                        )
            except Exception as e:
                logger.error(f"Error in global drift detection: {e}")
        
        return {
            "model_name": model_name,
            "model_version": model_version,
            "status": "success",
            "drift_detected": drift_detected,
            "drift_scores": drift_scores,
            "monitored_features": common_columns
        }
    
    def start_monitoring(self, interval_seconds: int = 300):
        """
        Start background monitoring thread.
        
        Args:
            interval_seconds: Interval between drift checks in seconds
        """
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started drift monitoring with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        logger.info("Stopping drift monitoring")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                # Get list of active model versions
                models_to_monitor = self._get_active_models()
                
                for model in models_to_monitor:
                    model_name = model["name"]
                    model_version = model["version"]
                    
                    logger.debug(f"Checking drift for {model_name}/{model_version}")
                    drift_result = self.detect_drift(model_name, model_version)
                    
                    if drift_result.get("drift_detected", False):
                        logger.warning(f"Drift detected for {model_name}/{model_version}")
            
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            # Sleep until next check
            time.sleep(interval_seconds)
    
    def _get_active_models(self) -> List[Dict[str, str]]:
        """
        Get list of active models to monitor.
        
        Returns:
            List of dictionaries with model name and version
        """
        # Implementation depends on deployment type
        # Sample implementation: read from config file
        try:
            if os.path.exists("config/monitoring/monitored_models.json"):
                with open("config/monitoring/monitored_models.json", "r") as f:
                    return json.load(f)
            else:
                return []
        except Exception as e:
            logger.error(f"Failed to get active models: {e}")
            return []
    
    def _send_drift_alert(
        self,
        model_name: str,
        model_version: str,
        drift_score: float,
        details: Dict[str, float]
    ) -> None:
        """
        Send drift alert notification.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            drift_score: Overall drift score
            details: Detailed drift scores by feature
        """
        # Implementation depends on notification system
        # Example: HTTP webhook
        try:
            import requests
            
            payload = {
                "model_name": model_name,
                "model_version": model_version,
                "drift_score": drift_score,
                "details": details,
                "timestamp": time.time(),
                "severity": "warning" if drift_score > self.drift_threshold else "info"
            }
            
            response = requests.post(
                self.notification_uri,
                json=payload,
                timeout=5
            )
            
            if response.status_code >= 400:
                logger.warning(f"Failed to send drift alert: {response.status_code} {response.text}")
        
        except Exception as e:
            logger.error(f"Failed to send drift alert: {e}")

# Set up FastAPI routes
drift_service = None

@app.on_event("startup")
async def startup_event():
    global drift_service
    
    # Get configuration from environment
    reference_data_store_uri = os.environ.get(
        "REFERENCE_DATA_STORE_URI",
        "postgres://postgres:postgres@postgres:5432/driftml"
    )
    production_data_store_uri = os.environ.get(
        "PRODUCTION_DATA_STORE_URI",
        "redis://redis:6379/0"
    )
    notification_uri = os.environ.get("NOTIFICATION_URI")
    
    # Start Prometheus metrics server
    metrics_port = int(os.environ.get("METRICS_PORT", "8081"))
    start_http_server(metrics_port)
    
    # Initialize drift detection service
    drift_service = DriftDetectionService(
        reference_data_store_uri=reference_data_store_uri,
        production_data_store_uri=production_data_store_uri,
        notification_uri=notification_uri
    )
    
    # Start background monitoring if enabled
    if os.environ.get("ENABLE_BACKGROUND_MONITORING", "false").lower() == "true":
        interval = int(os.environ.get("MONITORING_INTERVAL_SECONDS", "300"))
        drift_service.start_monitoring(interval)

@app.get("/")
async def root():
    return {"message": "Drift-ML Drift Detection Service"}

@app.post("/detect-drift/{model_name}/{model_version}")
async def detect_drift(model_name: str, model_version: str):
    if not drift_service:
        raise HTTPException(status_code=503, detail="Drift detection service not initialized")
    
    result = drift_service.detect_drift(model_name, model_version)
    return result

@app.post("/monitoring/start")
async def start_monitoring(interval_seconds: int = 300):
    if not drift_service:
        raise HTTPException(status_code=503, detail="Drift detection service not initialized")
    
    drift_service.start_monitoring(interval_seconds)
    return {"status": "success", "message": f"Started monitoring with {interval_seconds}s interval"}

@app.post("/monitoring/stop")
async def stop_monitoring():
    if not drift_service:
        raise HTTPException(status_code=503, detail="Drift detection service not initialized")
    
    drift_service.stop_monitoring()
    return {"status": "success", "message": "Stopped monitoring"}
