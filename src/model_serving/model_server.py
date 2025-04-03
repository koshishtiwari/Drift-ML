"""
Model Serving module for Drift-ML platform.
Provides functionality for deploying and serving ML models.
"""

import os
import json
import time
import yaml
import uuid
import requests
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from kubernetes import client, config
from loguru import logger
from fastapi import FastAPI, HTTPException, Request, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

# Import security module
from src.security.security import Security


class ModelDetails(BaseModel):
    """Pydantic model for model details."""
    name: str
    version: Optional[str] = None
    uri: Optional[str] = None
    stage: Optional[str] = "Production"
    description: Optional[str] = None


class InferenceRequest(BaseModel):
    """Pydantic model for inference request."""
    model_name: str
    model_version: Optional[str] = None
    model_stage: Optional[str] = "Production"
    inputs: Union[List[Dict[str, Any]], Dict[str, List[Any]], List[List[Any]]] = Field(...)


class InferenceResponse(BaseModel):
    """Pydantic model for inference response."""
    model_name: str
    model_version: str
    predictions: List[Any]
    prediction_time_ms: float
    metadata: Optional[Dict[str, Any]] = None


class ModelDeployer:
    """Base class for model deployment."""
    
    def __init__(self, model_registry_uri: Optional[str] = None):
        """
        Initialize the model deployer.
        
        Args:
            model_registry_uri: URI for MLflow model registry
        """
        self.model_registry_uri = model_registry_uri
        
        # Set up MLflow
        if self.model_registry_uri:
            mlflow.set_registry_uri(self.model_registry_uri)
        
        self.client = MlflowClient()
    
    def deploy(
        self,
        model_details: ModelDetails,
        config: Dict[str, Any],
        wait_for_deployment: bool = True,
        timeout_seconds: int = 300
    ) -> Dict[str, Any]:
        """
        Deploy a model (to be implemented by subclasses).
        
        Args:
            model_details: Details of the model to deploy
            config: Deployment configuration
            wait_for_deployment: Whether to wait for deployment to complete
            timeout_seconds: Timeout in seconds when waiting for deployment
            
        Returns:
            Deployment details
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def undeploy(
        self,
        deployment_name: str,
        namespace: Optional[str] = None
    ) -> bool:
        """
        Undeploy a model (to be implemented by subclasses).
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            
        Returns:
            True if successful, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def get_deployment_status(
        self,
        deployment_name: str,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get the status of a deployment (to be implemented by subclasses).
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            
        Returns:
            Deployment status details
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def list_deployments(
        self,
        namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List all deployments (to be implemented by subclasses).
        
        Args:
            namespace: Kubernetes namespace
            
        Returns:
            List of deployments
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def _get_model_uri(self, model_details: ModelDetails) -> str:
        """
        Get the URI for a model.
        
        Args:
            model_details: Model details
            
        Returns:
            Model URI
        """
        if model_details.uri:
            return model_details.uri
        
        name = model_details.name
        version = model_details.version
        stage = model_details.stage
        
        if version:
            return f"models:/{name}/{version}"
        elif stage:
            # Get latest version in the specified stage
            versions = self.client.get_latest_versions(name, stages=[stage])
            if not versions:
                raise ValueError(f"No model versions found for {name} in stage {stage}")
            
            version = versions[0].version
            return f"models:/{name}/{version}"
        else:
            # Get the latest version
            versions = self.client.get_latest_versions(name)
            if not versions:
                raise ValueError(f"No model versions found for {name}")
            
            version = max(versions, key=lambda x: int(x.version)).version
            return f"models:/{name}/{version}"


class KServeDeployer(ModelDeployer):
    """Model deployer for KServe on Kubernetes."""
    
    def __init__(
        self,
        model_registry_uri: Optional[str] = None,
        namespace: str = "default",
        kserve_api_version: str = "serving.kserve.io/v1beta1",
        security: Optional[Security] = None
    ):
        """
        Initialize the KServe deployer.
        
        Args:
            model_registry_uri: URI for MLflow model registry
            namespace: Kubernetes namespace
            kserve_api_version: KServe API version
            security: Security module instance
        """
        super().__init__(model_registry_uri)
        self.namespace = namespace
        self.api_version = kserve_api_version
        self.security = security
        
        # Set up Kubernetes client
        try:
            config.load_incluster_config()
            logger.info("Using in-cluster Kubernetes configuration")
        except:
            try:
                config.load_kube_config()
                logger.info("Using kubeconfig for Kubernetes configuration")
            except:
                logger.warning("Could not load Kubernetes configuration")
        
        self.k8s_client = client.CustomObjectsApi()
    
    def deploy(
        self,
        model_details: ModelDetails,
        config: Dict[str, Any],
        wait_for_deployment: bool = True,
        timeout_seconds: int = 300,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Deploy a model using KServe.
        
        Args:
            model_details: Details of the model to deploy
            config: Deployment configuration
            wait_for_deployment: Whether to wait for deployment to complete
            timeout_seconds: Timeout in seconds when waiting for deployment
            user_id: ID of the user performing the action
            
        Returns:
            Deployment details
        """
        # Check authorization if security module is available
        if self.security and user_id:
            has_permission = self.security.authz.check_permission(
                user_id=user_id,
                resource="model",
                action="deploy"
            )
            
            if not has_permission:
                logger.warning(f"User {user_id} does not have permission to deploy models")
                raise PermissionError("You do not have permission to deploy models")
            
            # Log the action in audit log
            self.security.audit.log_event(
                action="deploy_model",
                resource="model",
                resource_id=f"{model_details.name}/{model_details.version}",
                user_id=user_id,
                details={
                    "model_name": model_details.name,
                    "model_version": model_details.version,
                    "config": {k: v for k, v in config.items() if k not in ["service_account_token"]}
                }
            )
        
        model_uri = self._get_model_uri(model_details)
        deployment_name = config.get("name", f"{model_details.name.lower()}-{str(uuid.uuid4())[:8]}")
        namespace = config.get("namespace", self.namespace)
        
        # Create KServe InferenceService spec
        inference_service = {
            "apiVersion": self.api_version,
            "kind": "InferenceService",
            "metadata": {
                "name": deployment_name,
                "namespace": namespace,
                "annotations": {
                    "serving.kserve.io/deploymentMode": config.get("deployment_mode", "Serverless"),
                    "drift-ml/model-name": model_details.name,
                    "drift-ml/model-version": model_details.version or "latest",
                    "drift-ml/deployed-at": datetime.now().isoformat()
                }
            },
            "spec": {
                "predictor": {
                    "serviceAccountName": config.get("service_account", "default"),
                    "minReplicas": config.get("min_replicas", 1),
                    "maxReplicas": config.get("max_replicas", 3),
                    "containers": [
                        {
                            "name": "kserve-container",
                            "image": config.get("image", "kserve/mlserver:latest"),
                            "resources": {
                                "requests": {
                                    "cpu": config.get("cpu_request", "100m"),
                                    "memory": config.get("memory_request", "256Mi")
                                },
                                "limits": {
                                    "cpu": config.get("cpu_limit", "1"),
                                    "memory": config.get("memory_limit", "1Gi")
                                }
                            },
                            "env": [
                                {"name": "STORAGE_URI", "value": model_uri}
                            ]
                        }
                    ]
                }
            }
        }
        
        # Add model format if specified
        model_format = config.get("model_format")
        if model_format:
            inference_service["spec"]["predictor"]["model"] = {
                "modelFormat": model_format,
                "name": model_details.name,
                "storageUri": model_uri
            }
        
        try:
            # Create or update the InferenceService
            try:
                # Try to get existing service first
                self.k8s_client.get_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    name=deployment_name
                )
                
                # If it exists, update it
                response = self.k8s_client.replace_namespaced_custom_object(
                    group="serving.kserve.io",
                    version="v1beta1",
                    namespace=namespace,
                    plural="inferenceservices",
                    name=deployment_name,
                    body=inference_service
                )
                logger.info(f"Updated InferenceService {deployment_name} in namespace {namespace}")
            except client.rest.ApiException as e:
                if e.status == 404:
                    # If it doesn't exist, create it
                    response = self.k8s_client.create_namespaced_custom_object(
                        group="serving.kserve.io",
                        version="v1beta1",
                        namespace=namespace,
                        plural="inferenceservices",
                        body=inference_service
                    )
                    logger.info(f"Created InferenceService {deployment_name} in namespace {namespace}")
                else:
                    raise
            
            # Wait for deployment to be ready if requested
            if wait_for_deployment:
                is_ready = self._wait_for_deployment(deployment_name, namespace, timeout_seconds)
                if not is_ready:
                    logger.warning(f"Deployment {deployment_name} not ready within {timeout_seconds} seconds")
            
            # Get deployment details
            return self.get_deployment_status(deployment_name, namespace)
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            raise
    
    def undeploy(
        self,
        deployment_name: str,
        namespace: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> bool:
        """
        Undeploy a model from KServe.
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            user_id: ID of the user performing the action
            
        Returns:
            True if successful, False otherwise
        """
        # Check authorization if security module is available
        if self.security and user_id:
            has_permission = self.security.authz.check_permission(
                user_id=user_id,
                resource="model",
                action="undeploy"
            )
            
            if not has_permission:
                logger.warning(f"User {user_id} does not have permission to undeploy models")
                raise PermissionError("You do not have permission to undeploy models")
            
            # Log the action in audit log
            self.security.audit.log_event(
                action="undeploy_model",
                resource="model",
                resource_id=deployment_name,
                user_id=user_id,
                details={
                    "deployment_name": deployment_name,
                    "namespace": namespace or self.namespace
                }
            )
        
        namespace = namespace or self.namespace
        
        try:
            self.k8s_client.delete_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=deployment_name
            )
            logger.info(f"Deleted InferenceService {deployment_name} from namespace {namespace}")
            return True
        except Exception as e:
            logger.error(f"Failed to undeploy model: {e}")
            return False
    
    def get_deployment_status(
        self,
        deployment_name: str,
        namespace: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get the status of a KServe deployment.
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            user_id: ID of the user performing the action
            
        Returns:
            Deployment status details
        """
        # Check authorization if security module is available
        if self.security and user_id:
            has_permission = self.security.authz.check_permission(
                user_id=user_id,
                resource="model",
                action="view"
            )
            
            if not has_permission:
                logger.warning(f"User {user_id} does not have permission to view model deployments")
                raise PermissionError("You do not have permission to view model deployments")
        
        namespace = namespace or self.namespace
        
        try:
            response = self.k8s_client.get_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices",
                name=deployment_name
            )
            
            status = response.get("status", {})
            conditions = status.get("conditions", [])
            
            # Determine overall status
            is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
            
            # Extract URL if available
            url = None
            if "url" in status:
                url = status["url"]
            
            # Extract model details from annotations
            annotations = response["metadata"].get("annotations", {})
            model_name = annotations.get("drift-ml/model-name", "unknown")
            model_version = annotations.get("drift-ml/model-version", "unknown")
            deployed_at = annotations.get("drift-ml/deployed-at", "unknown")
            
            return {
                "name": deployment_name,
                "namespace": namespace,
                "model_name": model_name,
                "model_version": model_version,
                "deployed_at": deployed_at,
                "status": "Ready" if is_ready else "Not Ready",
                "conditions": conditions,
                "url": url,
                "raw_status": status
            }
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            return {
                "name": deployment_name,
                "namespace": namespace,
                "status": "Error",
                "error": str(e)
            }
    
    def list_deployments(
        self,
        namespace: Optional[str] = None,
        model_name: Optional[str] = None,
        user_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        List all KServe deployments.
        
        Args:
            namespace: Kubernetes namespace
            model_name: Filter by model name
            user_id: ID of the user performing the action
            
        Returns:
            List of deployments
        """
        # Check authorization if security module is available
        if self.security and user_id:
            has_permission = self.security.authz.check_permission(
                user_id=user_id,
                resource="model",
                action="view"
            )
            
            if not has_permission:
                logger.warning(f"User {user_id} does not have permission to view model deployments")
                raise PermissionError("You do not have permission to view model deployments")
        
        namespace = namespace or self.namespace
        
        try:
            response = self.k8s_client.list_namespaced_custom_object(
                group="serving.kserve.io",
                version="v1beta1",
                namespace=namespace,
                plural="inferenceservices"
            )
            
            deployments = []
            for item in response.get("items", []):
                annotations = item["metadata"].get("annotations", {})
                deployment_model_name = annotations.get("drift-ml/model-name", "unknown")
                
                # Filter by model name if specified
                if model_name and deployment_model_name != model_name:
                    continue
                
                deployment_name = item["metadata"]["name"]
                status = item.get("status", {})
                conditions = status.get("conditions", [])
                is_ready = any(c.get("type") == "Ready" and c.get("status") == "True" for c in conditions)
                
                url = status.get("url")
                model_version = annotations.get("drift-ml/model-version", "unknown")
                deployed_at = annotations.get("drift-ml/deployed-at", "unknown")
                
                deployments.append({
                    "name": deployment_name,
                    "namespace": namespace,
                    "model_name": deployment_model_name,
                    "model_version": model_version,
                    "deployed_at": deployed_at,
                    "status": "Ready" if is_ready else "Not Ready",
                    "url": url
                })
            
            return deployments
        except Exception as e:
            logger.error(f"Failed to list deployments: {e}")
            return []
    
    def _wait_for_deployment(
        self,
        deployment_name: str,
        namespace: str,
        timeout_seconds: int = 300
    ) -> bool:
        """
        Wait for a deployment to be ready.
        
        Args:
            deployment_name: Name of the deployment
            namespace: Kubernetes namespace
            timeout_seconds: Timeout in seconds
            
        Returns:
            True if deployment is ready, False if timed out
        """
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            status = self.get_deployment_status(deployment_name, namespace)
            if status["status"] == "Ready":
                return True
            
            # Wait a bit before checking again
            time.sleep(5)
        
        return False


class LocalModelServer:
    """
    Local model server for development and testing purposes.
    Uses FastAPI to serve models directly.
    """
    
    def __init__(
        self,
        model_registry_uri: Optional[str] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
        security: Optional[Security] = None
    ):
        """
        Initialize the local model server.
        
        Args:
            model_registry_uri: URI for MLflow model registry
            host: Host to bind the server to
            port: Port to bind the server to
            security: Security module instance
        """
        self.model_registry_uri = model_registry_uri
        self.host = host
        self.port = port
        self.security = security
        
        # Set up MLflow
        if self.model_registry_uri:
            mlflow.set_registry_uri(self.model_registry_uri)
        
        self.client = MlflowClient()
        
        # Map to store loaded models
        self.models = {}
        
        # Create FastAPI app
        self.app = FastAPI(title="Drift-ML Local Model Server")
        
        # Set up security if provided
        self.token_auth = HTTPBearer(auto_error=False)
        
        self._setup_routes()
    
    async def _get_current_user(
        self, 
        credentials: Optional[HTTPAuthorizationCredentials] = Security(HTTPBearer(auto_error=False))
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current user from JWT token.
        
        Args:
            credentials: HTTP Authorization credentials
            
        Returns:
            User information or None if authentication fails
        """
        if not self.security or not credentials:
            return None
        
        token = credentials.credentials
        try:
            payload = self.security.auth.verify_token(token)
            return payload
        except Exception as e:
            logger.error(f"Token verification failed: {e}")
            return None
    
    async def _check_permission(
        self, 
        user: Optional[Dict[str, Any]], 
        resource: str, 
        action: str
    ) -> bool:
        """
        Check if a user has permission for a resource action.
        
        Args:
            user: User information
            resource: Resource to access
            action: Action to perform
            
        Returns:
            True if permitted, False otherwise
        """
        if not self.security or not user:
            return False
        
        try:
            user_id = int(user.get("sub"))
            return self.security.authz.check_permission(user_id, resource, action)
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    def _setup_routes(self) -> None:
        """Set up API routes."""
        
        @self.app.get("/")
        async def root():
            return {"message": "Drift-ML Local Model Server"}
        
        @self.app.get("/models")
        async def list_models(request: Request, user: Dict = Depends(self._get_current_user)):
            # Check permission if security is enabled
            if self.security:
                if not user:
                    raise HTTPException(status_code=401, detail="Not authenticated")
                
                has_permission = await self._check_permission(user, "model", "view")
                if not has_permission:
                    raise HTTPException(status_code=403, detail="Not authorized to view models")
                
                # Log audit event
                try:
                    user_id = int(user.get("sub"))
                    self.security.audit.log_event(
                        action="list_models",
                        resource="model",
                        user_id=user_id,
                        username=user.get("username"),
                        ip_address=request.client.host
                    )
                except Exception as e:
                    logger.error(f"Failed to log audit event: {e}")
            
            return {"models": list(self.models.keys())}
        
        @self.app.post("/models")
        async def load_model(
            model_details: ModelDetails, 
            request: Request, 
            user: Dict = Depends(self._get_current_user)
        ):
            # Check permission if security is enabled
            if self.security:
                if not user:
                    raise HTTPException(status_code=401, detail="Not authenticated")
                
                has_permission = await self._check_permission(user, "model", "deploy")
                if not has_permission:
                    raise HTTPException(status_code=403, detail="Not authorized to load models")
            
            try:
                model_uri = self._get_model_uri(model_details)
                model_key = f"{model_details.name}/{model_details.version or 'latest'}"
                
                # Load model
                model = mlflow.pyfunc.load_model(model_uri)
                
                # Store model with metadata
                self.models[model_key] = {
                    "model": model,
                    "name": model_details.name,
                    "version": model_details.version or "latest",
                    "uri": model_uri,
                    "loaded_at": datetime.now().isoformat()
                }
                
                # Log audit event if security is enabled
                if self.security and user:
                    try:
                        user_id = int(user.get("sub"))
                        self.security.audit.log_event(
                            action="load_model",
                            resource="model",
                            resource_id=model_key,
                            user_id=user_id,
                            username=user.get("username"),
                            ip_address=request.client.host,
                            details={
                                "model_name": model_details.name,
                                "model_version": model_details.version,
                                "model_uri": model_uri
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to log audit event: {e}")
                
                return {
                    "status": "success",
                    "message": f"Model {model_key} loaded successfully",
                    "model_key": model_key
                }
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.delete("/models/{model_name}/{model_version}")
        async def unload_model(
            model_name: str, 
            model_version: str = "latest", 
            request: Request, 
            user: Dict = Depends(self._get_current_user)
        ):
            # Check permission if security is enabled
            if self.security:
                if not user:
                    raise HTTPException(status_code=401, detail="Not authenticated")
                
                has_permission = await self._check_permission(user, "model", "undeploy")
                if not has_permission:
                    raise HTTPException(status_code=403, detail="Not authorized to unload models")
            
            model_key = f"{model_name}/{model_version}"
            if model_key in self.models:
                del self.models[model_key]
                
                # Log audit event if security is enabled
                if self.security and user:
                    try:
                        user_id = int(user.get("sub"))
                        self.security.audit.log_event(
                            action="unload_model",
                            resource="model",
                            resource_id=model_key,
                            user_id=user_id,
                            username=user.get("username"),
                            ip_address=request.client.host
                        )
                    except Exception as e:
                        logger.error(f"Failed to log audit event: {e}")
                
                return {"status": "success", "message": f"Model {model_key} unloaded"}
            else:
                raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
        
        @self.app.post("/predict")
        async def predict(
            request: InferenceRequest, 
            req: Request, 
            user: Dict = Depends(self._get_current_user)
        ):
            # Check permission if security is enabled
            if self.security:
                if not user:
                    raise HTTPException(status_code=401, detail="Not authenticated")
                
                has_permission = await self._check_permission(user, "model", "predict")
                if not has_permission:
                    raise HTTPException(status_code=403, detail="Not authorized to make predictions")
            
            model_key = f"{request.model_name}/{request.model_version or 'latest'}"
            
            # If model stage is specified, override the version
            if request.model_stage and not request.model_version:
                try:
                    # Get latest version in the specified stage
                    versions = self.client.get_latest_versions(request.model_name, stages=[request.model_stage])
                    if versions:
                        model_version = versions[0].version
                        model_key = f"{request.model_name}/{model_version}"
                except:
                    # If failed to get model by stage, continue with the original key
                    pass
            
            if model_key not in self.models:
                # Try to load the model automatically
                try:
                    model_details = ModelDetails(
                        name=request.model_name,
                        version=request.model_version,
                        stage=request.model_stage
                    )
                    model_uri = self._get_model_uri(model_details)
                    
                    # Load model
                    model = mlflow.pyfunc.load_model(model_uri)
                    
                    # Store model with metadata
                    self.models[model_key] = {
                        "model": model,
                        "name": request.model_name,
                        "version": request.model_version or "latest",
                        "uri": model_uri,
                        "loaded_at": datetime.now().isoformat()
                    }
                    
                    # Log audit event if security is enabled
                    if self.security and user:
                        try:
                            user_id = int(user.get("sub"))
                            self.security.audit.log_event(
                                action="auto_load_model",
                                resource="model",
                                resource_id=model_key,
                                user_id=user_id,
                                username=user.get("username"),
                                ip_address=req.client.host,
                                details={
                                    "model_name": request.model_name,
                                    "model_version": request.model_version,
                                    "model_stage": request.model_stage,
                                    "auto_loaded": True
                                }
                            )
                        except Exception as e:
                            logger.error(f"Failed to log audit event: {e}")
                    
                    logger.info(f"Auto-loaded model {model_key}")
                except Exception as e:
                    logger.error(f"Failed to auto-load model: {e}")
                    raise HTTPException(
                        status_code=404,
                        detail=f"Model {model_key} not found and could not be auto-loaded: {str(e)}"
                    )
            
            # Get the model
            model_info = self.models[model_key]
            model = model_info["model"]
            
            # Parse input data
            try:
                if isinstance(request.inputs, list) and all(isinstance(x, dict) for x in request.inputs):
                    # List of dictionaries (records format)
                    data = pd.DataFrame.from_records(request.inputs)
                elif isinstance(request.inputs, dict) and all(isinstance(x, list) for x in request.inputs.values()):
                    # Dictionary of lists (column format)
                    data = pd.DataFrame(request.inputs)
                elif isinstance(request.inputs, list) and all(isinstance(x, list) for x in request.inputs):
                    # List of lists (values format)
                    data = pd.DataFrame(request.inputs)
                else:
                    # Try to handle it as raw input
                    data = request.inputs
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid input format: {str(e)}"
                )
            
            # Make prediction
            start_time = time.time()
            try:
                predictions = model.predict(data)
                
                # Convert numpy arrays to lists for JSON serialization
                if isinstance(predictions, np.ndarray):
                    predictions = predictions.tolist()
                
                prediction_time_ms = (time.time() - start_time) * 1000
                
                # Log audit event if security is enabled
                if self.security and user:
                    try:
                        user_id = int(user.get("sub"))
                        self.security.audit.log_event(
                            action="predict",
                            resource="model",
                            resource_id=model_key,
                            user_id=user_id,
                            username=user.get("username"),
                            ip_address=req.client.host,
                            details={
                                "model_name": model_info["name"],
                                "model_version": model_info["version"],
                                "prediction_time_ms": prediction_time_ms,
                                "input_shape": getattr(data, "shape", None),
                                "prediction_count": len(predictions) if isinstance(predictions, list) else 1
                            }
                        )
                    except Exception as e:
                        logger.error(f"Failed to log audit event: {e}")
                
                return InferenceResponse(
                    model_name=model_info["name"],
                    model_version=model_info["version"],
                    predictions=predictions,
                    prediction_time_ms=prediction_time_ms,
                    metadata={
                        "model_uri": model_info["uri"],
                        "loaded_at": model_info["loaded_at"]
                    }
                )
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Prediction error: {str(e)}"
                )
    
    def start(self) -> None:
        """Start the server."""
        import uvicorn
        logger.info(f"Starting local model server on {self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)
    
    def _get_model_uri(self, model_details: ModelDetails) -> str:
        """
        Get the URI for a model.
        
        Args:
            model_details: Model details
            
        Returns:
            Model URI
        """
        if model_details.uri:
            return model_details.uri
        
        name = model_details.name
        version = model_details.version
        stage = model_details.stage
        
        if version:
            return f"models:/{name}/{version}"
        elif stage:
            # Get latest version in the specified stage
            versions = self.client.get_latest_versions(name, stages=[stage])
            if not versions:
                raise ValueError(f"No model versions found for {name} in stage {stage}")
            
            version = versions[0].version
            return f"models:/{name}/{version}"
        else:
            # Get the latest version
            versions = self.client.get_latest_versions(name)
            if not versions:
                raise ValueError(f"No model versions found for {name}")
            
            version = max(versions, key=lambda x: int(x.version)).version
            return f"models:/{name}/{version}"


class ModelClient:
    """Client for interacting with model servers."""
    
    def __init__(
        self,
        server_url: str,
        auth_token: Optional[str] = None
    ):
        """
        Initialize the model client.
        
        Args:
            server_url: URL of the model server
            auth_token: Authentication token for the server
        """
        self.server_url = server_url.rstrip("/")
        self.auth_token = auth_token
        self.headers = {}
        
        if self.auth_token:
            self.headers["Authorization"] = f"Bearer {self.auth_token}"
    
    def predict(
        self,
        model_name: str,
        inputs: Union[List[Dict[str, Any]], Dict[str, List[Any]], List[List[Any]]],
        model_version: Optional[str] = None,
        model_stage: Optional[str] = "Production",
        timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """
        Make a prediction request to the model server.
        
        Args:
            model_name: Name of the model
            inputs: Input data for prediction
            model_version: Specific model version to use
            model_stage: Model stage to use if version not specified
            timeout_seconds: Request timeout in seconds
            
        Returns:
            Prediction results
        """
        url = f"{self.server_url}/predict"
        
        payload = {
            "model_name": model_name,
            "inputs": inputs
        }
        
        if model_version:
            payload["model_version"] = model_version
        
        if model_stage:
            payload["model_stage"] = model_stage
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.headers,
                timeout=timeout_seconds
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                error_message = response.text
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail", response.text)
                except:
                    pass
                
                raise Exception(f"Prediction request failed with status {response.status_code}: {error_message}")
        except requests.Timeout:
            raise Exception(f"Prediction request timed out after {timeout_seconds} seconds")
        except Exception as e:
            raise Exception(f"Prediction request failed: {str(e)}")
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models on the server.
        
        Returns:
            List of available models
        """
        url = f"{self.server_url}/models"
        
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json().get("models", [])
            else:
                error_message = response.text
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail", response.text)
                except:
                    pass
                
                raise Exception(f"List models request failed with status {response.status_code}: {error_message}")
        except Exception as e:
            raise Exception(f"List models request failed: {str(e)}")
    
    def load_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
        model_stage: Optional[str] = "Production"
    ) -> Dict[str, Any]:
        """
        Request the server to load a model.
        
        Args:
            model_name: Name of the model
            model_version: Specific model version to load
            model_stage: Model stage to load if version not specified
            
        Returns:
            Response details
        """
        url = f"{self.server_url}/models"
        
        payload = {
            "name": model_name
        }
        
        if model_version:
            payload["version"] = model_version
        
        if model_stage:
            payload["stage"] = model_stage
        
        try:
            response = requests.post(url, json=payload, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_message = response.text
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail", response.text)
                except:
                    pass
                
                raise Exception(f"Load model request failed with status {response.status_code}: {error_message}")
        except Exception as e:
            raise Exception(f"Load model request failed: {str(e)}")
    
    def unload_model(
        self,
        model_name: str,
        model_version: Optional[str] = "latest"
    ) -> Dict[str, Any]:
        """
        Request the server to unload a model.
        
        Args:
            model_name: Name of the model
            model_version: Model version to unload
            
        Returns:
            Response details
        """
        version = model_version or "latest"
        url = f"{self.server_url}/models/{model_name}/{version}"
        
        try:
            response = requests.delete(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                error_message = response.text
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail", response.text)
                except:
                    pass
                
                raise Exception(f"Unload model request failed with status {response.status_code}: {error_message}")
        except Exception as e:
            raise Exception(f"Unload model request failed: {str(e)}")


# Kubernetes YAML manifests for model deployment
KSERVE_INFERENCE_SERVICE_TEMPLATE = """
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: {name}
  namespace: {namespace}
  annotations:
    serving.kserve.io/deploymentMode: {deployment_mode}
    drift-ml/model-name: {model_name}
    drift-ml/model-version: {model_version}
    drift-ml/deployed-at: {deployed_at}
spec:
  predictor:
    serviceAccountName: {service_account}
    minReplicas: {min_replicas}
    maxReplicas: {max_replicas}
    containers:
    - name: kserve-container
      image: {image}
      resources:
        requests:
          cpu: {cpu_request}
          memory: {memory_request}
        limits:
          cpu: {cpu_limit}
          memory: {memory_limit}
      env:
      - name: STORAGE_URI
        value: {model_uri}
"""

# Example usage
if __name__ == "__main__":
    # Load security module
    from src.security.security import Security
    
    security = Security(
        db_url="sqlite:///security.db",
        jwt_secret="your-jwt-secret-key"
    )
    
    # Set up default roles and permissions
    security.setup_default_roles_and_permissions()
    
    # Create admin user
    admin_id = security.create_initial_admin_user(
        username="admin",
        email="admin@example.com",
        password="secure-password"
    )
    
    # Start a local model server with security
    server = LocalModelServer(
        model_registry_uri="sqlite:///mlflow.db",
        host="0.0.0.0",
        port=8080,
        security=security
    )
    
    # Get auth token for API requests
    auth_info = security.auth.authenticate_user(
        username="admin",
        password="secure-password"
    )
    
    if auth_info:
        token = security.auth.generate_token(
            user_id=auth_info["id"],
            username=auth_info["username"],
            is_admin=auth_info["is_admin"]
        )
        
        # Create a test client to interact with the server
        client = ModelClient(
            server_url="http://localhost:8080",
            auth_token=token
        )
        
        # Load model (this will be handled by the server)
        try:
            response = client.load_model(
                model_name="example_model",
                model_version="1"
            )
            
            print(f"Loaded model: {response}")
            
            # Make a prediction
            inputs = [
                {"feature1": 0.5, "feature2": 0.2, "feature3": 0.3},
                {"feature1": 0.1, "feature2": 0.8, "feature3": 0.7}
            ]
            
            prediction = client.predict(
                model_name="example_model",
                model_version="1",
                inputs=inputs
            )
            
            print(f"Prediction: {prediction}")
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Start the server (this will block)
    server.start()