"""
Monitoring module for Drift-ML platform.
Provides functionality for tracking system health and model performance.
"""
import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import threading
import numpy as np
import pandas as pd
from loguru import logger
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary, CollectorRegistry, push_to_gateway
from prometheus_client.exposition import generate_latest

class MetricDefinition(BaseModel):
    """Pydantic model for metric definition."""
    name: str
    description: str
    type: str = "counter"  # counter, gauge, histogram, summary
    labels: List[str] = []
    buckets: Optional[List[float]] = None  # for histogram
    quantiles: Optional[List[float]] = None  # for summary

class ModelMetrics:
    """
    Manages metrics for ML models.
    Provides Prometheus metrics for model predictions, latency, etc.
    """
    
    def __init__(
        self,
        model_name: str,
        model_version: str,
        registry: Optional[CollectorRegistry] = None,
        push_gateway_url: Optional[str] = None,
        push_interval_seconds: int = 10
    ):
        """
        Initialize model metrics.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            registry: Prometheus registry
            push_gateway_url: URL for Prometheus push gateway
            push_interval_seconds: Interval for pushing metrics to gateway
        """
        self.model_name = model_name
        self.model_version = model_version
        self.registry = registry or prometheus_client.REGISTRY
        self.push_gateway_url = push_gateway_url
        self.push_interval_seconds = push_interval_seconds
        
        # Basic metric labels
        self.base_labels = {
            "model_name": model_name,
            "model_version": model_version
        }
        
        # Initialize metrics
        self._init_metrics()
        
        # Start push thread if a gateway is configured
        if self.push_gateway_url:
            self._start_pusher()
    
    def _init_metrics(self) -> None:
        """Initialize metrics for model monitoring."""
        # Prediction metrics
        self.prediction_counter = Counter(
            "model_predictions_total",
            "Total number of predictions",
            ["model_name", "model_version", "status"],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            "model_prediction_latency_seconds",
            "Prediction latency in seconds",
            ["model_name", "model_version"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
            registry=self.registry
        )
        
        # Feature metrics
        self.feature_value = Gauge(
            "model_feature_value",
            "Feature value statistics",
            ["model_name", "model_version", "feature_name", "stat"],
            registry=self.registry
        )
        
        # Prediction distribution metrics
        self.prediction_value = Gauge(
            "model_prediction_value",
            "Prediction value statistics",
            ["model_name", "model_version", "stat"],
            registry=self.registry
        )
        
        # Model drift metrics
        self.drift_score = Gauge(
            "model_drift_score",
            "Model drift score",
            ["model_name", "model_version", "metric"],
            registry=self.registry
        )
        
        # System resource metrics
        self.memory_usage = Gauge(
            "model_server_memory_bytes",
            "Memory usage in bytes",
            ["model_name", "model_version"],
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            "model_server_cpu_utilization",
            "CPU utilization percentage",
            ["model_name", "model_version"],
            registry=self.registry
        )
    
    def _start_pusher(self) -> None:
        """Start a thread to periodically push metrics to the gateway."""
        def push_metrics() -> None:
            while True:
                try:
                    prometheus_client.push_to_gateway(
                        self.push_gateway_url,
                        job=f"drift_ml_{self.model_name}_{self.model_version}",
                        registry=self.registry
                    )
                    logger.debug(f"Pushed metrics to gateway: {self.push_gateway_url}")
                except Exception as e:
                    logger.error(f"Failed to push metrics to gateway: {e}")
                
                time.sleep(self.push_interval_seconds)
        
        thread = threading.Thread(target=push_metrics, daemon=True)
        thread.start()
    
    def record_prediction(
        self,
        latency_seconds: float,
        status: str = "success",
        features: Optional[Union[pd.DataFrame, Dict[str, Any], List[Any]]] = None,
        predictions: Optional[Union[List[Any], np.ndarray]] = None
    ) -> None:
        """
        Record metrics for a prediction.
        
        Args:
            latency_seconds: Prediction latency in seconds
            status: Status of the prediction (success, error)
            features: Input features for the prediction
            predictions: Prediction outputs
        """
        # Increment prediction counter
        self.prediction_counter.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            status=status
        ).inc()
        
        # Record prediction latency
        self.prediction_latency.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).observe(latency_seconds)
        
        # Record feature statistics if available
        if features is not None:
            self._record_feature_stats(features)
        
        # Record prediction statistics if available
        if predictions is not None:
            self._record_prediction_stats(predictions)
    
    def _record_feature_stats(
        self,
        features: Union[pd.DataFrame, Dict[str, Any], List[Any]]
    ) -> None:
        """
        Record feature statistics.
        
        Args:
            features: Features to record statistics for
        """
        # Convert to DataFrame if necessary
        if not isinstance(features, pd.DataFrame):
            if isinstance(features, dict):
                features_df = pd.DataFrame(features, index=[0])
            elif isinstance(features, list):
                if all(isinstance(x, dict) for x in features):
                    features_df = pd.DataFrame(features)
                else:
                    features_df = pd.DataFrame([features])
            else:
                logger.warning(f"Unsupported feature type: {type(features)}")
                return
        else:
            features_df = features
        
        # Calculate and record statistics for each numeric feature
        for column in features_df.select_dtypes(include=np.number).columns:
            values = features_df[column].dropna()
            
            if len(values) == 0:
                continue
            
            # Record basic statistics
            self.feature_value.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                feature_name=column,
                stat="mean"
            ).set(values.mean())
            
            self.feature_value.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                feature_name=column,
                stat="min"
            ).set(values.min())
            
            self.feature_value.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                feature_name=column,
                stat="max"
            ).set(values.max())
            
            # Calculate percentiles
            for p in [50, 90, 95, 99]:
                self.feature_value.labels(
                    model_name=self.model_name,
                    model_version=self.model_version,
                    feature_name=column,
                    stat=f"p{p}"
                ).set(np.percentile(values, p))
    
    def _record_prediction_stats(
        self,
        predictions: Union[List[Any], np.ndarray]
    ) -> None:
        """
        Record prediction statistics.
        
        Args:
            predictions: Predictions to record statistics for
        """
        # Convert to numpy array if necessary
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        
        # Skip if not numeric
        if not np.issubdtype(predictions.dtype, np.number):
            return
        
        # Record basic statistics
        self.prediction_value.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            stat="mean"
        ).set(np.mean(predictions))
        
        self.prediction_value.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            stat="min"
        ).set(np.min(predictions))
        
        self.prediction_value.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            stat="max"
        ).set(np.max(predictions))
        
        # Calculate percentiles
        for p in [50, 90, 95, 99]:
            self.prediction_value.labels(
                model_name=self.model_name,
                model_version=self.model_version,
                stat=f"p{p}"
            ).set(np.percentile(predictions, p))
    
    def record_drift_score(
        self,
        metric: str,
        score: float
    ) -> None:
        """
        Record a model drift score.
        
        Args:
            metric: Name of the drift metric
            score: Drift score value
        """
        self.drift_score.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            metric=metric
        ).set(score)
    
    def record_system_metrics(
        self,
        memory_bytes: Optional[float] = None,
        cpu_percent: Optional[float] = None
    ) -> None:
        """
        Record system resource metrics.
        
        Args:
            memory_bytes: Memory usage in bytes
            cpu_percent: CPU utilization percentage
        """
        if memory_bytes is not None:
            self.memory_usage.labels(
                model_name=self.model_name,
                model_version=self.model_version
            ).set(memory_bytes)
        
        if cpu_percent is not None:
            self.cpu_usage.labels(
                model_name=self.model_name,
                model_version=self.model_version
            ).set(cpu_percent)

class MetricsMiddleware:
    """
    Middleware for collecting metrics from FastAPI endpoints.
    Can be used with the model server to collect metrics.
    """
    
    def __init__(
        self,
        registry: Optional[CollectorRegistry] = None,
    ):
        """
        Initialize the metrics middleware.
        
        Args:
            registry: Prometheus registry
        """
        self.registry = registry or prometheus_client.REGISTRY
        
        # Initialize metrics
        self.request_counter = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry
        )
        
        self.request_latency = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds",
            ["method", "endpoint"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0),
            registry=self.registry
        )
        
        self.request_size = Histogram(
            "http_request_size_bytes",
            "HTTP request size in bytes",
            ["method", "endpoint"],
            buckets=(10, 100, 1000, 10000, 100000, 1000000),
            registry=self.registry
        )
        
        self.response_size = Histogram(
            "http_response_size_bytes",
            "HTTP response size in bytes",
            ["method", "endpoint", "status"],
            buckets=(10, 100, 1000, 10000, 100000, 1000000),
            registry=self.registry
        )
    
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process the request and record metrics.
        
        Args:
            request: FastAPI request
            call_next: Next middleware in the chain
            
        Returns:
            Response from the next middleware
        """
        # Record request size
        request_body = await request.body()
        request_size = len(request_body)
        
        # Reset request body
        request._body = request_body
        
        # Get endpoint for path parameters
        route = request.scope.get("route")
        endpoint = request.url.path
        if route and route.path != endpoint:
            endpoint = route.path
        
        # Record request size
        self.request_size.labels(
            method=request.method,
            endpoint=endpoint
        ).observe(request_size)
        
        # Start timer for latency tracking
        start_time = time.time()
        
        # Process request
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            raise e
        finally:
            # Record latency
            latency = time.time() - start_time
            self.request_latency.labels(
                method=request.method,
                endpoint=endpoint
            ).observe(latency)
            
            # Record request counter
            self.request_counter.labels(
                method=request.method,
                endpoint=endpoint,
                status=status
            ).inc()
        
        # Capture and record response size
        original_response_body = response.body
        response_size = len(original_response_body)
        
        self.response_size.labels(
            method=request.method,
            endpoint=endpoint,
            status=status
        ).observe(response_size)
        
        return response

class MonitoringService:
    """
    Service for model monitoring and metrics collection.
    """
    
    def __init__(
        self,
        push_gateway_url: Optional[str] = None,
        push_interval_seconds: int = 10
    ):
        """
        Initialize the monitoring service.
        
        Args:
            push_gateway_url: URL for Prometheus push gateway
            push_interval_seconds: Interval for pushing metrics to gateway
        """
        self.registry = CollectorRegistry()
        self.push_gateway_url = push_gateway_url
        self.push_interval_seconds = push_interval_seconds
        
        # Store model metrics instances
        self.model_metrics: Dict[str, ModelMetrics] = {}
        
        # Create FastAPI app for metrics
        self.app = FastAPI(title="Drift-ML Monitoring Service")
        self._setup_routes()
    
    def _setup_routes(self) -> None:
        """Set up API routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "Drift-ML Monitoring Service"}
        
        @self.app.get("/metrics")
        async def metrics():
            """Expose Prometheus metrics."""
            return Response(
                content=generate_latest(self.registry),
                media_type="text/plain"
            )
    
    def get_model_metrics(
        self,
        model_name: str,
        model_version: str
    ) -> ModelMetrics:
        """
        Get or create metrics for a specific model.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            
        Returns:
            ModelMetrics instance
        """
        model_key = f"{model_name}/{model_version}"
        
        if model_key not in self.model_metrics:
            self.model_metrics[model_key] = ModelMetrics(
                model_name=model_name,
                model_version=model_version,
                registry=self.registry,
                push_gateway_url=self.push_gateway_url,
                push_interval_seconds=self.push_interval_seconds
            )
        
        return self.model_metrics[model_key]
    
    def get_metrics_middleware(self) -> MetricsMiddleware:
        """
        Get a metrics middleware for FastAPI.
        
        Returns:
            MetricsMiddleware instance
        """
        return MetricsMiddleware(registry=self.registry)
    
    def start(
        self,
        host: str = "0.0.0.0",
        port: int = 8081
    ) -> None:
        """
        Start the monitoring service.
        
        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        import uvicorn
        logger.info(f"Starting monitoring service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port)

# Example usage
if __name__ == "__main__":
    # Create a monitoring service
    monitoring_service = MonitoringService(
        push_gateway_url="http://localhost:9091"
    )
    
    # Get metrics for a model
    model_metrics = monitoring_service.get_model_metrics(
        model_name="example_model",
        model_version="1"
    )
    
    # Record a prediction
    model_metrics.record_prediction(
        latency_seconds=0.1,
        status="success",
        features={"feature1": [0.5], "feature2": [0.2]},
        predictions=[0.8, 0.9]
    )
    
    # Record a drift score
    model_metrics.record_drift_score(
        metric="psi",
        score=0.05
    )
    
    # Start the monitoring service (this will block)
    monitoring_service.start()