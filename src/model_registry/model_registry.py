"""
Model Registry module for Drift-ML platform.
Provides functionality for storing, versioning, and managing trained models.
"""
import os
import json
import tempfile
import shutil
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from loguru import logger

class ModelRegistry:
    """MLflow-based model registry for ML models."""
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        default_tags: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the model registry.
        
        Args:
            tracking_uri: MLflow tracking URI
            registry_uri: MLflow registry URI (if different from tracking URI)
            default_tags: Default tags to apply to registered models
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)
        
        self.client = MlflowClient()
        self.default_tags = default_tags or {}
    
    def register_model(
        self,
        run_id: str,
        model_name: str,
        model_path: str = "model",
        version_description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Register a model from an MLflow run.
        
        Args:
            run_id: ID of the MLflow run containing the model
            model_name: Name to register the model under
            model_path: Path to the model in the MLflow run
            version_description: Description for the model version
            tags: Tags to apply to the model version
            
        Returns:
            Model version as a string
        """
        try:
            # Register the model
            result = mlflow.register_model(
                model_uri=f"runs:/{run_id}/{model_path}",
                name=model_name
            )
            
            model_version = result.version
            logger.info(f"Registered model '{model_name}' version {model_version}")
            
            # Add description if provided
            if version_description:
                self.client.update_model_version(
                    name=model_name,
                    version=model_version,
                    description=version_description
                )
            
            # Add tags
            all_tags = {**self.default_tags, **(tags or {})}
            for key, value in all_tags.items():
                self.client.set_model_version_tag(
                    name=model_name,
                    version=model_version,
                    key=key,
                    value=value
                )
            
            return model_version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def create_model(
        self,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Create a new registered model.
        
        Args:
            model_name: Name for the registered model
            description: Description for the model
            tags: Tags to apply to the model
        """
        try:
            # Create the model
            self.client.create_registered_model(
                name=model_name,
                description=description,
                tags={**self.default_tags, **(tags or {})}
            )
            
            logger.info(f"Created registered model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            raise
    
    def update_model(
        self,
        model_name: str,
        description: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Update a registered model.
        
        Args:
            model_name: Name of the registered model
            description: New description for the model
            tags: Tags to apply to the model
        """
        try:
            # Update model description if provided
            if description is not None:
                self.client.update_registered_model(
                    name=model_name,
                    description=description
                )
            
            # Update tags if provided
            if tags is not None:
                for key, value in tags.items():
                    self.client.set_registered_model_tag(
                        name=model_name,
                        key=key,
                        value=value
                    )
            
            logger.info(f"Updated registered model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to update model: {e}")
            raise
    
    def transition_model_version_stage(
        self,
        model_name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ) -> None:
        """
        Transition a model version to a different stage.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            stage: Stage to transition to (e.g., "Staging", "Production")
            archive_existing_versions: Whether to archive existing versions in the target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            
            logger.info(f"Transitioned model '{model_name}' version {version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model version: {e}")
            raise
    
    def get_latest_version(
        self,
        model_name: str,
        stages: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the registered model
            stages: List of stages to filter by
            
        Returns:
            Dictionary with model version information or None if not found
        """
        try:
            if stages is not None:
                versions = []
                for stage in stages:
                    versions.extend(
                        self.client.get_latest_versions(model_name, stages=[stage])
                    )
            else:
                versions = self.client.get_latest_versions(model_name)
            
            if not versions:
                return None
            
            # Sort by version number (descending)
            versions.sort(key=lambda x: int(x.version), reverse=True)
            latest = versions[0]
            
            return {
                "name": latest.name,
                "version": latest.version,
                "stage": latest.current_stage,
                "description": latest.description,
                "run_id": latest.run_id,
                "creation_timestamp": latest.creation_timestamp,
                "last_updated_timestamp": latest.last_updated_timestamp,
                "user_id": latest.user_id,
                "status": latest.status,
                "source": latest.source
            }
        except Exception as e:
            logger.error(f"Failed to get latest version: {e}")
            return None
    
    def get_model_version(
        self,
        model_name: str,
        version: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            
        Returns:
            Dictionary with model version information or None if not found
        """
        try:
            version_obj = self.client.get_model_version(
                name=model_name,
                version=version
            )
            
            return {
                "name": version_obj.name,
                "version": version_obj.version,
                "stage": version_obj.current_stage,
                "description": version_obj.description,
                "run_id": version_obj.run_id,
                "creation_timestamp": version_obj.creation_timestamp,
                "last_updated_timestamp": version_obj.last_updated_timestamp,
                "user_id": version_obj.user_id,
                "status": version_obj.status,
                "source": version_obj.source
            }
        except Exception as e:
            logger.error(f"Failed to get model version: {e}")
            return None
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models.
        
        Returns:
            List of model information dictionaries
        """
        try:
            models = self.client.list_registered_models()
            
            return [{
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "last_updated_timestamp": model.last_updated_timestamp,
                "description": model.description,
                "latest_versions": [
                    {
                        "version": v.version,
                        "stage": v.current_stage
                    }
                    for v in model.latest_versions
                ]
            } for model in models]
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def list_model_versions(
        self,
        model_name: str,
        stages: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        List versions of a registered model.
        
        Args:
            model_name: Name of the registered model
            stages: List of stages to filter by
            
        Returns:
            List of model version information dictionaries
        """
        try:
            # Get all versions for the model
            versions = self.client.search_model_versions(f"name='{model_name}'")
            
            # Filter by stage if specified
            if stages is not None:
                versions = [v for v in versions if v.current_stage in stages]
            
            # Sort by version number (descending)
            versions.sort(key=lambda x: int(x.version), reverse=True)
            
            return [{
                "name": v.name,
                "version": v.version,
                "stage": v.current_stage,
                "description": v.description,
                "run_id": v.run_id,
                "creation_timestamp": v.creation_timestamp,
                "last_updated_timestamp": v.last_updated_timestamp,
                "user_id": v.user_id,
                "status": v.status,
                "source": v.source
            } for v in versions]
        except Exception as e:
            logger.error(f"Failed to list model versions: {e}")
            return []
    
    def add_model_version_tag(
        self,
        model_name: str,
        version: str,
        key: str,
        value: str
    ) -> None:
        """
        Add a tag to a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            key: Tag key
            value: Tag value
        """
        try:
            self.client.set_model_version_tag(
                name=model_name,
                version=version,
                key=key,
                value=value
            )
            
            logger.info(f"Added tag {key}={value} to model '{model_name}' version {version}")
        except Exception as e:
            logger.error(f"Failed to add model version tag: {e}")
            raise
    
    def add_model_tag(
        self,
        model_name: str,
        key: str,
        value: str
    ) -> None:
        """
        Add a tag to a registered model.
        
        Args:
            model_name: Name of the registered model
            key: Tag key
            value: Tag value
        """
        try:
            self.client.set_registered_model_tag(
                name=model_name,
                key=key,
                value=value
            )
            
            logger.info(f"Added tag {key}={value} to model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to add model tag: {e}")
            raise
    
    def delete_model(
        self,
        model_name: str
    ) -> None:
        """
        Delete a registered model.
        
        Args:
            model_name: Name of the registered model
        """
        try:
            self.client.delete_registered_model(name=model_name)
            
            logger.info(f"Deleted model '{model_name}'")
        except Exception as e:
            logger.error(f"Failed to delete model: {e}")
            raise
    
    def delete_model_version(
        self,
        model_name: str,
        version: str
    ) -> None:
        """
        Delete a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
        """
        try:
            self.client.delete_model_version(
                name=model_name,
                version=version
            )
            
            logger.info(f"Deleted model '{model_name}' version {version}")
        except Exception as e:
            logger.error(f"Failed to delete model version: {e}")
            raise
    
    def download_model(
        self,
        model_name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
        dst_path: Optional[str] = None
    ) -> str:
        """
        Download a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model (if not provided, use stage)
            stage: Stage of the model (if version not provided)
            dst_path: Destination path
            
        Returns:
            Path to the downloaded model
        """
        try:
            if version is not None:
                model_uri = f"models:/{model_name}/{version}"
            elif stage is not None:
                model_uri = f"models:/{model_name}/{stage}"
            else:
                # Get latest version
                latest = self.get_latest_version(model_name)
                if latest is None:
                    raise ValueError(f"No versions found for model '{model_name}'")
                model_uri = f"models:/{model_name}/{latest['version']}"
            
            if dst_path is None:
                dst_path = tempfile.mkdtemp()
            
            # Download the model
            path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=dst_path)
            
            logger.info(f"Downloaded model to {path}")
            
            return path
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise
    
    def get_model_dependencies(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, List[str]]:
        """
        Get dependencies for a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            
        Returns:
            Dictionary of dependencies
        """
        try:
            # Get the run ID for the model version
            model_version = self.get_model_version(model_name, version)
            if model_version is None:
                raise ValueError(f"Model version {version} not found for model '{model_name}'")
            
            run_id = model_version["run_id"]
            
            # Get the run and its details
            run = self.client.get_run(run_id)
            params = run.data.params
            
            # Extract dependencies from params
            dependencies = {}
            
            # Check if conda environment is available
            artifacts_path = os.path.join("runs", run_id, "artifacts")
            if os.path.exists(os.path.join(artifacts_path, "conda.yaml")):
                with open(os.path.join(artifacts_path, "conda.yaml"), "r") as f:
                    conda_env = f.read()
                dependencies["conda_environment"] = conda_env
            
            # Check if requirements.txt is available
            if os.path.exists(os.path.join(artifacts_path, "requirements.txt")):
                with open(os.path.join(artifacts_path, "requirements.txt"), "r") as f:
                    requirements = f.read().splitlines()
                dependencies["python_requirements"] = requirements
            
            # Extract any other dependencies from params
            if "dependencies" in params:
                try:
                    deps = json.loads(params["dependencies"])
                    dependencies.update(deps)
                except json.JSONDecodeError:
                    dependencies["raw_dependencies"] = params["dependencies"]
            
            return dependencies
        except Exception as e:
            logger.error(f"Failed to get model dependencies: {e}")
            return {}
    
    def get_model_lineage(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, Any]:
        """
        Get model lineage information.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            
        Returns:
            Dictionary of lineage information
        """
        try:
            # Get the run ID for the model version
            model_version = self.get_model_version(model_name, version)
            if model_version is None:
                raise ValueError(f"Model version {version} not found for model '{model_name}'")
            
            run_id = model_version["run_id"]
            
            # Get the run and its details
            run = self.client.get_run(run_id)
            params = run.data.params
            tags = run.data.tags
            
            # Build lineage dictionary
            lineage = {
                "run_id": run_id,
                "parameters": params,
                "tags": tags,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "status": run.info.status
            }
            
            # Get dataset information if available
            if "dataset_id" in params:
                lineage["dataset_id"] = params["dataset_id"]
            
            # Get feature set information if available
            if "feature_set_id" in params:
                lineage["feature_set_id"] = params["feature_set_id"]
            
            # Get parent runs if available
            if "mlflow.parentRunId" in tags:
                parent_run_id = tags["mlflow.parentRunId"]
                lineage["parent_run_id"] = parent_run_id
                
                try:
                    parent_run = self.client.get_run(parent_run_id)
                    lineage["parent_run"] = {
                        "start_time": parent_run.info.start_time,
                        "end_time": parent_run.info.end_time,
                        "status": parent_run.info.status
                    }
                except Exception:
                    pass
            
            return lineage
        except Exception as e:
            logger.error(f"Failed to get model lineage: {e}")
            return {}
    
    def approve_model_version(
        self,
        model_name: str,
        version: str,
        approver: str,
        comment: Optional[str] = None,
        user_id: Optional[int] = None,
        auth_token: Optional[str] = None
    ) -> None:
        """
        Approve a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            approver: Name of the approver
            comment: Approval comment
            user_id: ID of the user performing the approval
            auth_token: Authentication token for the user
            
        Raises:
            PermissionError: If the user is not authorized to approve models
            ValueError: If model version does not exist
        """
        try:
            # Check if model version exists
            model_version = self.get_model_version(model_name, version)
            if model_version is None:
                raise ValueError(f"Model version {version} not found for model '{model_name}'")
            
            # Verify authorization if security module is available
            if hasattr(self, 'security') and self.security:
                if user_id is None and auth_token is None:
                    raise PermissionError("Authentication required for model approval")
                
                # Verify via user_id
                if user_id is not None:
                    has_permission = self.security.authz.check_permission(
                        user_id=user_id,
                        resource="model",
                        action="approve"
                    )
                    if not has_permission:
                        raise PermissionError(f"User ID {user_id} does not have permission to approve models")
                
                # Verify via token
                elif auth_token is not None:
                    token_data = self.security.auth.validate_token(auth_token)
                    if not token_data:
                        raise PermissionError("Invalid authentication token")
                    
                    token_user_id = token_data.get("sub")
                    if token_user_id:
                        has_permission = self.security.authz.check_permission(
                            user_id=int(token_user_id),
                            resource="model",
                            action="approve"
                        )
                        if not has_permission:
                            raise PermissionError("Token does not have permission to approve models")
                        
                        # Override approver with authenticated username if available
                        if token_data.get("username"):
                            approver = token_data.get("username")
            
            # Add approval tags
            self.add_model_version_tag(
                model_name=model_name,
                version=version,
                key="approved",
                value="true"
            )
            
            self.add_model_version_tag(
                model_name=model_name,
                version=version,
                key="approved_by",
                value=approver
            )
            
            self.add_model_version_tag(
                model_name=model_name,
                version=version,
                key="approved_at",
                value=datetime.utcnow().isoformat()
            )
            
            if comment:
                self.add_model_version_tag(
                    model_name=model_name,
                    version=version,
                    key="approval_comment",
                    value=comment
                )
            
            # Audit log
            if hasattr(self, 'security') and self.security:
                self.security.audit.log_event(
                    action="approve_model",
                    resource="model",
                    resource_id=f"{model_name}/{version}",
                    user_id=user_id,
                    details={
                        "model_name": model_name,
                        "model_version": version,
                        "approver": approver,
                        "comment": comment
                    }
                )
            
            logger.info(f"Approved model '{model_name}' version {version} by {approver}")
        except (ValueError, PermissionError) as e:
            # Re-raise expected errors
            logger.error(f"Model approval error: {e}")
            raise
        except Exception as e:
            # For unexpected errors, provide context but still raise
            logger.error(f"Unexpected error during model approval: {e}")
            raise RuntimeError(f"Failed to approve model version: {e}") from e
    
    def reject_model_version(
        self,
        model_name: str,
        version: str,
        rejector: str,
        reason: Optional[str] = None
    ) -> None:
        """
        Reject a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            rejector: Name of the rejector
            reason: Rejection reason
        """
        try:
            # Check if model version exists
            model_version = self.get_model_version(model_name, version)
            if model_version is None:
                raise ValueError(f"Model version {version} not found for model '{model_name}'")
            
            # Add rejection tags
            self.add_model_version_tag(
                model_name=model_name,
                version=version,
                key="rejected",
                value="true"
            )
            
            self.add_model_version_tag(
                model_name=model_name,
                version=version,
                key="rejected_by",
                value=rejector
            )
            
            self.add_model_version_tag(
                model_name=model_name,
                version=version,
                key="rejected_at",
                value=datetime.utcnow().isoformat()
            )
            
            if reason:
                self.add_model_version_tag(
                    model_name=model_name,
                    version=version,
                    key="rejection_reason",
                    value=reason
                )
            
            logger.info(f"Rejected model '{model_name}' version {version} by {rejector}")
        except Exception as e:
            logger.error(f"Failed to reject model version: {e}")
            raise
    
    def get_approval_status(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, Any]:
        """
        Get approval status of a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            
        Returns:
            Dictionary of approval status information
        """
        try:
            # Get the tags for the model version
            version_details = self.client.get_model_version(
                name=model_name,
                version=version
            )
            
            # Get tags
            tags = {}
            for tag in self.client.get_model_version_tags(model_name, version):
                tags[tag.key] = tag.value
            
            status = {
                "stage": version_details.current_stage,
                "status": version_details.status
            }
            
            # Check if approved
            if "approved" in tags and tags["approved"] == "true":
                status["approved"] = True
                status["approved_by"] = tags.get("approved_by")
                status["approved_at"] = tags.get("approved_at")
                status["approval_comment"] = tags.get("approval_comment")
            elif "rejected" in tags and tags["rejected"] == "true":
                status["approved"] = False
                status["rejected_by"] = tags.get("rejected_by")
                status["rejected_at"] = tags.get("rejected_at")
                status["rejection_reason"] = tags.get("rejection_reason")
            else:
                status["approved"] = None  # Not yet approved or rejected
            
            return status
        except Exception as e:
            logger.error(f"Failed to get approval status: {e}")
            return {"approved": None, "error": str(e)}
    
    def get_model_metrics(
        self,
        model_name: str,
        version: str
    ) -> Dict[str, float]:
        """
        Get metrics for a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            
        Returns:
            Dictionary of metrics
        """
        try:
            # Get the run ID for the model version
            model_version = self.get_model_version(model_name, version)
            if model_version is None:
                raise ValueError(f"Model version {version} not found for model '{model_name}'")
            
            run_id = model_version["run_id"]
            
            # Get the run and its metrics
            run = self.client.get_run(run_id)
            metrics = run.data.metrics
            
            return metrics
        except Exception as e:
            logger.error(f"Failed to get model metrics: {e}")
            return {}
    
    def compare_models(
        self,
        model_name: str,
        versions: List[str],
        metric_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple versions of a model.
        
        Args:
            model_name: Name of the registered model
            versions: List of model versions to compare
            metric_names: List of metrics to compare (if None, compare all)
            
        Returns:
            DataFrame comparing the model versions
            
        Raises:
            ValueError: If the model name does not exist
            ValueError: If any specified version does not exist
        """
        if not versions:
            raise ValueError("No model versions specified for comparison")
            
        try:
            # Verify model exists
            model = self.client.get_registered_model(model_name)
            if not model:
                raise ValueError(f"Model '{model_name}' does not exist")
                
            # Verify all versions exist
            for version in versions:
                model_version = self.get_model_version(model_name, version)
                if model_version is None:
                    raise ValueError(f"Version {version} of model '{model_name}' does not exist")
            
            all_metrics = {}
            all_params = {}
            all_tags = {}
            
            for version in versions:
                # Get metrics
                metrics = self.get_model_metrics(model_name, version)
                all_metrics[version] = metrics
                
                # Get parameters
                model_version = self.get_model_version(model_name, version)
                run_id = model_version["run_id"]
                run = self.client.get_run(run_id)
                all_params[version] = run.data.params
                
                # Get tags
                tags = {}
                for tag in self.client.get_model_version_tags(model_name, version):
                    tags[tag.key] = tag.value
                all_tags[version] = tags
            
            # Create comparison dataframe
            comparison = {}
            
            # Add metrics
            if metric_names is not None:
                for metric in metric_names:
                    for version in versions:
                        key = f"{metric} (v{version})"
                        comparison[key] = all_metrics.get(version, {}).get(metric)
            else:
                # Add all metrics
                all_metric_names = set()
                for metrics in all_metrics.values():
                    all_metric_names.update(metrics.keys())
                
                for metric in all_metric_names:
                    for version in versions:
                        key = f"{metric} (v{version})"
                        comparison[key] = all_metrics.get(version, {}).get(metric)
            
            # Add important parameters
            important_params = ["model_type", "hyperparams"]
            for param in important_params:
                for version in versions:
                    key = f"{param} (v{version})"
                    comparison[key] = all_params.get(version, {}).get(param)
            
            # Add approval status
            for version in versions:
                key = f"approval_status (v{version})"
                approved = all_tags.get(version, {}).get("approved") == "true"
                rejected = all_tags.get(version, {}).get("rejected") == "true"
                
                if approved:
                    comparison[key] = "Approved"
                elif rejected:
                    comparison[key] = "Rejected"
                else:
                    comparison[key] = "Pending"
            
            # Create DataFrame
            df = pd.DataFrame([comparison])
            
            return df
            
        except ValueError as e:
            # Re-raise ValueError for expected error conditions
            logger.error(f"Model comparison error: {e}")
            raise
        except Exception as e:
            # For unexpected errors, provide context but still raise
            logger.error(f"Unexpected error during model comparison: {e}")
            raise RuntimeError(f"Failed to compare models: {e}") from e
    
    def log_deployment_event(
        self,
        model_name: str,
        version: str,
        environment: str,
        status: str,
        user: str,
        deployment_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log a model deployment event.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            environment: Deployment environment
            status: Deployment status
            user: User who initiated the deployment
            deployment_id: Unique deployment ID
            details: Additional deployment details
        """
        try:
            # Generate deployment ID if not provided
            if deployment_id is None:
                deployment_id = f"deploy_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
            
            # Create deployment event tag
            event = {
                "environment": environment,
                "status": status,
                "user": user,
                "timestamp": datetime.utcnow().isoformat(),
                "details": details or {}
            }
            
            # Add the deployment event as a tag
            self.add_model_version_tag(
                model_name=model_name,
                version=version,
                key=f"deployment_{deployment_id}",
                value=json.dumps(event)
            )
            
            # Add latest deployment status tag
            self.add_model_version_tag(
                model_name=model_name,
                version=version,
                key=f"latest_deployment_{environment}",
                value=status
            )
            
            logger.info(f"Logged deployment event for model '{model_name}' version {version} in {environment}")
        except Exception as e:
            logger.error(f"Failed to log deployment event: {e}")
            raise
    
    def get_deployment_history(
        self,
        model_name: str,
        version: str
    ) -> List[Dict[str, Any]]:
        """
        Get deployment history for a model version.
        
        Args:
            model_name: Name of the registered model
            version: Version of the model
            
        Returns:
            List of deployment events
        """
        try:
            # Get all tags for the model version
            tags = {}
            for tag in self.client.get_model_version_tags(model_name, version):
                tags[tag.key] = tag.value
            
            # Filter deployment events
            deployment_events = []
            for key, value in tags.items():
                if key.startswith("deployment_"):
                    try:
                        event = json.loads(value)
                        event["id"] = key.replace("deployment_", "")
                        deployment_events.append(event)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid deployment event format for tag {key}")
            
            # Sort by timestamp descending
            deployment_events.sort(
                key=lambda x: x.get("timestamp", ""),
                reverse=True
            )
            
            return deployment_events
        except Exception as e:
            logger.error(f"Failed to get deployment history: {e}")
            return []

# Example usage
if __name__ == "__main__":
    # Initialize model registry
    registry = ModelRegistry(
        tracking_uri="http://localhost:5000",
        default_tags={"owner": "drift-ml"}
    )
    
    # Register a model from a run
    version = registry.register_model(
        run_id="run_id",
        model_name="example_model",
        version_description="Initial model version"
    )
    
    # Transition the model to staging
    registry.transition_model_version_stage(
        model_name="example_model",
        version=version,
        stage="Staging"
    )
    
    # Approve the model
    registry.approve_model_version(
        model_name="example_model",
        version=version,
        approver="data_scientist",
        comment="Model meets performance criteria"
    )
    
    # Transition the model to production
    registry.transition_model_version_stage(
        model_name="example_model",
        version=version,
        stage="Production",
        archive_existing_versions=True
    )
    
    # Log deployment event
    registry.log_deployment_event(
        model_name="example_model",
        version=version,
        environment="production",
        status="deployed",
        user="mlops_engineer",
        details={"instance_type": "ml.c5.xlarge"}
    )
    
    # Get deployment history
    deployments = registry.get_deployment_history(
        model_name="example_model",
        version=version
    )
    print(f"Deployment history: {deployments}")