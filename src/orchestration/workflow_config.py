"""
Workflow configuration module for Drift-ML platform.
Provides functionality for managing workflow configurations.
"""
import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from loguru import logger

class WorkflowConfig:
    """Class representing a workflow configuration."""
    
    def __init__(
        self,
        name: str,
        description: str,
        tasks: List[Dict[str, Any]],
        schedule: Optional[str] = None,
        version: str = "1.0.0",
        created_by: str = "drift-ml",
        created_at: Optional[datetime] = None,
        updated_at: Optional[datetime] = None,
        enabled: bool = True,
        tags: Optional[List[str]] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a workflow configuration.
        
        Args:
            name: Workflow name
            description: Workflow description
            tasks: List of task configurations
            schedule: Cron schedule expression
            version: Configuration version
            created_by: Creator of the workflow
            created_at: Creation timestamp
            updated_at: Last update timestamp
            enabled: Whether the workflow is enabled
            tags: Tags for categorizing the workflow
            params: Default parameters for the workflow
        """
        self.name = name
        self.description = description
        self.tasks = tasks
        self.schedule = schedule
        self.version = version
        self.created_by = created_by
        self.created_at = created_at or datetime.utcnow()
        self.updated_at = updated_at or datetime.utcnow()
        self.enabled = enabled
        self.tags = tags or []
        self.params = params or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow configuration to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "tasks": self.tasks,
            "schedule": self.schedule,
            "version": self.version,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "enabled": self.enabled,
            "tags": self.tags,
            "params": self.params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowConfig":
        """Create workflow configuration from dictionary."""
        created_at = data.get("created_at")
        if created_at and isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        
        updated_at = data.get("updated_at")
        if updated_at and isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        
        return cls(
            name=data["name"],
            description=data["description"],
            tasks=data["tasks"],
            schedule=data.get("schedule"),
            version=data.get("version", "1.0.0"),
            created_by=data.get("created_by", "drift-ml"),
            created_at=created_at,
            updated_at=updated_at,
            enabled=data.get("enabled", True),
            tags=data.get("tags", []),
            params=data.get("params", {})
        )

class WorkflowConfigManager:
    """Manager for workflow configurations."""
    
    def __init__(self, config_dir: str):
        """
        Initialize the workflow configuration manager.
        
        Args:
            config_dir: Directory to store workflow configurations
        """
        self.config_dir = config_dir
        os.makedirs(config_dir, exist_ok=True)
    
    def _get_config_path(self, name: str) -> str:
        """
        Get file path for a workflow configuration.
        
        Args:
            name: Workflow name
            
        Returns:
            Path to the configuration file
        """
        return os.path.join(self.config_dir, f"{name}.json")
    
    def save_config(self, config: WorkflowConfig) -> None:
        """
        Save a workflow configuration.
        
        Args:
            config: Workflow configuration to save
        """
        config_path = self._get_config_path(config.name)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            
            logger.info(f"Saved workflow configuration for {config.name} to {config_path}")
        except Exception as e:
            logger.error(f"Failed to save workflow configuration: {e}")
            raise
    
    def get_config(self, name: str) -> Optional[WorkflowConfig]:
        """
        Get a workflow configuration.
        
        Args:
            name: Workflow name
            
        Returns:
            Workflow configuration or None if not found
        """
        config_path = self._get_config_path(name)
        
        if not os.path.exists(config_path):
            return None
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            return WorkflowConfig.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load workflow configuration: {e}")
            return None
    
    def list_configs(self, tag: Optional[str] = None) -> List[str]:
        """
        List available workflow configurations.
        
        Args:
            tag: Filter configurations by tag
            
        Returns:
            List of workflow names
        """
        configs = []
        
        try:
            for filename in os.listdir(self.config_dir):
                if filename.endswith(".json"):
                    name = filename[:-5]  # Remove .json extension
                    
                    if tag:
                        # Check if config has the specified tag
                        config = self.get_config(name)
                        if config and tag in config.tags:
                            configs.append(name)
                    else:
                        configs.append(name)
        except Exception as e:
            logger.error(f"Failed to list workflow configurations: {e}")
        
        return configs
    
    def delete_config(self, name: str) -> bool:
        """
        Delete a workflow configuration.
        
        Args:
            name: Workflow name
            
        Returns:
            True if deleted successfully, False otherwise
        """
        config_path = self._get_config_path(name)
        
        if not os.path.exists(config_path):
            return False
        
        try:
            os.remove(config_path)
            logger.info(f"Deleted workflow configuration for {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete workflow configuration: {e}")
            return False
    
    def update_config(self, name: str, updates: Dict[str, Any]) -> Optional[WorkflowConfig]:
        """
        Update a workflow configuration.
        
        Args:
            name: Workflow name
            updates: Dictionary of updates to apply
            
        Returns:
            Updated workflow configuration or None if not found
        """
        config = self.get_config(name)
        
        if not config:
            return None
        
        # Create a dictionary representation of the config
        config_dict = config.to_dict()
        
        # Update fields
        for key, value in updates.items():
            if key in config_dict:
                config_dict[key] = value
        
        # Update timestamp
        config_dict["updated_at"] = datetime.utcnow().isoformat()
        
        # Create new config from updated dictionary
        updated_config = WorkflowConfig.from_dict(config_dict)
        
        # Save updated config
        self.save_config(updated_config)
        
        return updated_config

# Template configurations for common workflows
class WorkflowTemplates:
    """Standard workflow templates."""
    
    @staticmethod
    def feature_engineering() -> WorkflowConfig:
        """
        Feature engineering workflow template.
        
        Returns:
            Workflow configuration
        """
        return WorkflowConfig(
            name="feature_engineering",
            description="Compute and store features from raw data",
            tasks=[
                {
                    "id": "check_data_sources",
                    "type": "python",
                    "description": "Check data source availability",
                    "function": "drift_ml.tasks.data.check_sources",
                    "params": {},
                    "upstream": []
                },
                {
                    "id": "extract_data",
                    "type": "python",
                    "description": "Extract data from sources",
                    "function": "drift_ml.tasks.data.extract_data",
                    "params": {
                        "batch_size": 1000,
                        "timeout": 60
                    },
                    "upstream": ["check_data_sources"]
                },
                {
                    "id": "compute_features",
                    "type": "python",
                    "description": "Compute features from data",
                    "function": "drift_ml.tasks.features.compute_features",
                    "params": {},
                    "upstream": ["extract_data"]
                },
                {
                    "id": "store_features",
                    "type": "python",
                    "description": "Store features in feature store",
                    "function": "drift_ml.tasks.features.store_features",
                    "params": {},
                    "upstream": ["compute_features"]
                },
                {
                    "id": "validate_features",
                    "type": "python",
                    "description": "Validate stored features",
                    "function": "drift_ml.tasks.features.validate_features",
                    "params": {},
                    "upstream": ["store_features"]
                }
            ],
            schedule="0 */6 * * *",  # Every 6 hours
            tags=["feature-engineering", "data-processing"]
        )
    
    @staticmethod
    def model_training() -> WorkflowConfig:
        """
        Model training workflow template.
        
        Returns:
            Workflow configuration
        """
        return WorkflowConfig(
            name="model_training",
            description="Train and evaluate ML models",
            tasks=[
                {
                    "id": "prepare_training_data",
                    "type": "python",
                    "description": "Prepare training data from feature store",
                    "function": "drift_ml.tasks.training.prepare_data",
                    "params": {},
                    "upstream": []
                },
                {
                    "id": "train_model",
                    "type": "python",
                    "description": "Train model on prepared data",
                    "function": "drift_ml.tasks.training.train_model",
                    "params": {
                        "model_type": "classifier",
                        "hyperparams": {
                            "learning_rate": 0.01,
                            "max_depth": 5
                        }
                    },
                    "upstream": ["prepare_training_data"]
                },
                {
                    "id": "evaluate_model",
                    "type": "python",
                    "description": "Evaluate model performance",
                    "function": "drift_ml.tasks.training.evaluate_model",
                    "params": {
                        "metrics": ["accuracy", "precision", "recall", "f1"]
                    },
                    "upstream": ["train_model"]
                },
                {
                    "id": "register_model",
                    "type": "python",
                    "description": "Register model in model registry",
                    "function": "drift_ml.tasks.registry.register_model",
                    "params": {},
                    "upstream": ["evaluate_model"]
                }
            ],
            schedule="0 0 * * *",  # Daily at midnight
            tags=["model-training", "machine-learning"]
        )
    
    @staticmethod
    def model_deployment() -> WorkflowConfig:
        """
        Model deployment workflow template.
        
        Returns:
            Workflow configuration
        """
        return WorkflowConfig(
            name="model_deployment",
            description="Deploy model to serving environment",
            tasks=[
                {
                    "id": "get_model_from_registry",
                    "type": "python",
                    "description": "Get model from registry",
                    "function": "drift_ml.tasks.registry.get_model",
                    "params": {
                        "stage": "production"
                    },
                    "upstream": []
                },
                {
                    "id": "validate_model",
                    "type": "python",
                    "description": "Validate model before deployment",
                    "function": "drift_ml.tasks.serving.validate_model",
                    "params": {},
                    "upstream": ["get_model_from_registry"]
                },
                {
                    "id": "deploy_model",
                    "type": "python",
                    "description": "Deploy model to serving environment",
                    "function": "drift_ml.tasks.serving.deploy_model",
                    "params": {
                        "deployment_type": "kserve",
                        "replicas": 2
                    },
                    "upstream": ["validate_model"]
                },
                {
                    "id": "run_smoke_tests",
                    "type": "python",
                    "description": "Run smoke tests on deployed model",
                    "function": "drift_ml.tasks.serving.smoke_test",
                    "params": {},
                    "upstream": ["deploy_model"]
                },
                {
                    "id": "update_model_status",
                    "type": "python",
                    "description": "Update model status in registry",
                    "function": "drift_ml.tasks.registry.update_status",
                    "params": {},
                    "upstream": ["run_smoke_tests"]
                }
            ],
            schedule=None,  # Triggered manually
            tags=["model-deployment", "serving"]
        )
    
    @staticmethod
    def data_drift_detection() -> WorkflowConfig:
        """
        Data drift detection workflow template.
        
        Returns:
            Workflow configuration
        """
        return WorkflowConfig(
            name="data_drift_detection",
            description="Detect drift in input data distributions",
            tasks=[
                {
                    "id": "extract_production_data",
                    "type": "python",
                    "description": "Extract recent production data",
                    "function": "drift_ml.tasks.monitoring.extract_production_data",
                    "params": {
                        "lookback_hours": 24
                    },
                    "upstream": []
                },
                {
                    "id": "extract_reference_data",
                    "type": "python",
                    "description": "Extract reference data",
                    "function": "drift_ml.tasks.monitoring.extract_reference_data",
                    "params": {},
                    "upstream": []
                },
                {
                    "id": "compute_drift_metrics",
                    "type": "python",
                    "description": "Compute drift metrics",
                    "function": "drift_ml.tasks.monitoring.compute_drift_metrics",
                    "params": {
                        "metrics": ["ks_test", "js_divergence", "population_stability_index"]
                    },
                    "upstream": ["extract_production_data", "extract_reference_data"]
                },
                {
                    "id": "store_drift_metrics",
                    "type": "python",
                    "description": "Store drift metrics",
                    "function": "drift_ml.tasks.monitoring.store_drift_metrics",
                    "params": {},
                    "upstream": ["compute_drift_metrics"]
                },
                {
                    "id": "check_drift_thresholds",
                    "type": "python",
                    "description": "Check if drift exceeds thresholds",
                    "function": "drift_ml.tasks.monitoring.check_drift_thresholds",
                    "params": {
                        "thresholds": {
                            "ks_test": 0.1,
                            "js_divergence": 0.15,
                            "population_stability_index": 0.25
                        }
                    },
                    "upstream": ["store_drift_metrics"]
                },
                {
                    "id": "trigger_model_retraining",
                    "type": "python",
                    "description": "Trigger model retraining if needed",
                    "function": "drift_ml.tasks.orchestration.trigger_dag",
                    "params": {
                        "dag_id": "model_training"
                    },
                    "upstream": ["check_drift_thresholds"]
                }
            ],
            schedule="0 */12 * * *",  # Every 12 hours
            tags=["monitoring", "drift-detection"]
        )

# Example usage
if __name__ == "__main__":
    # Initialize config manager
    config_manager = WorkflowConfigManager("./workflow_configs")
    
    # Create templates
    feature_eng_config = WorkflowTemplates.feature_engineering()
    model_training_config = WorkflowTemplates.model_training()
    model_deployment_config = WorkflowTemplates.model_deployment()
    drift_detection_config = WorkflowTemplates.data_drift_detection()
    
    # Save configurations
    config_manager.save_config(feature_eng_config)
    config_manager.save_config(model_training_config)
    config_manager.save_config(model_deployment_config)
    config_manager.save_config(drift_detection_config)
    
    # List configurations
    configs = config_manager.list_configs()
    print(f"Available configurations: {configs}")
    
    # Filter by tag
    ml_configs = config_manager.list_configs(tag="machine-learning")
    print(f"Machine learning configurations: {ml_configs}")