"""
Model Training module for Drift-ML platform.
Provides functionality for training and evaluating ML models using LLMs and MLflow.
"""
import os
import json
import tempfile
import pickle
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
import ollama
import google.generativeai as genai
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger

class LLMIntegration:
    """Integration with LLMs for model training assistance."""
    
    def __init__(
        self,
        mode: str = "offline",
        model_name: str = "llama2",
        api_key: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize the LLM integration.
        
        Args:
            mode: Mode of operation ("offline" for ollama, "online" for Google Gemini)
            model_name: Model name to use (ollama model or gemini model name)
            api_key: API key for Google Gemini (not needed for ollama)
            temperature: Sampling temperature
        """
        self.mode = mode
        self.model_name = model_name
        self.temperature = temperature
        
        if mode == "online" and api_key:
            # Setup Google Gemini
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
        elif mode == "offline":
            # Ollama doesn't need setup
            pass
        else:
            raise ValueError("Invalid mode or missing API key for online mode")
    
    def query(self, prompt: str) -> str:
        """
        Send a query to the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            LLM response
        """
        try:
            if self.mode == "online":
                # Use Google Gemini
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": self.temperature}
                )
                return response.text
            else:
                # Use Ollama
                response = ollama.chat(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": self.temperature}
                )
                return response["message"]["content"]
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            return f"Error: {str(e)}"
    
    def suggest_model(self, dataset_description: str, task_type: str) -> Dict[str, Any]:
        """
        Get model architecture suggestions for a dataset.
        
        Args:
            dataset_description: Description of the dataset
            task_type: Type of ML task (classification, regression, etc.)
            
        Returns:
            Dictionary with model suggestions
        """
        prompt = f"""
        As an ML expert, suggest model architectures for the following dataset and task:
        
        Dataset Description: {dataset_description}
        Task Type: {task_type}
        
        Provide suggestions in the following JSON format:
        
        {{
            "recommended_models": [
                {{
                    "model_type": "string",  // e.g., "logistic_regression", "random_forest", "neural_network"
                    "rationale": "string",
                    "hyperparameters": {{
                        "param1": value,
                        "param2": value
                    }}
                }}
            ],
            "feature_engineering": [
                {{
                    "technique": "string",
                    "rationale": "string"
                }}
            ]
        }}
        
        ONLY return the JSON object, no additional text.
        """
        
        response = self.query(prompt)
        
        try:
            # Try to parse as JSON
            suggestions = json.loads(response)
            return suggestions
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            # Return a simple default
            return {
                "recommended_models": [
                    {
                        "model_type": "random_forest",
                        "rationale": "Good general-purpose model with minimal tuning required",
                        "hyperparameters": {
                            "n_estimators": 100,
                            "max_depth": 10
                        }
                    }
                ],
                "feature_engineering": []
            }
    
    def explain_results(self, metrics: Dict[str, float], model_type: str) -> str:
        """
        Get explanation of model performance.
        
        Args:
            metrics: Dictionary of evaluation metrics
            model_type: Type of model evaluated
            
        Returns:
            Explanation of results
        """
        prompt = f"""
        As an ML expert, explain the following model evaluation results:
        
        Model Type: {model_type}
        Metrics: {json.dumps(metrics, indent=2)}
        
        Provide a concise explanation of the model's performance, strengths, and weaknesses.
        Also suggest possible improvements.
        """
        
        return self.query(prompt)
    
    def suggest_preprocessing(self, data_sample: str, target_column: str) -> List[Dict[str, Any]]:
        """
        Get preprocessing suggestions for a dataset.
        
        Args:
            data_sample: Sample of the dataset (first few rows as string)
            target_column: Name of the target column
            
        Returns:
            List of preprocessing steps
        """
        prompt = f"""
        As an ML preprocessing expert, suggest preprocessing steps for the following dataset sample:
        
        Data Sample:
        {data_sample}
        
        Target Column: {target_column}
        
        Provide suggestions in the following JSON format:
        
        [
            {{
                "column": "string",
                "preprocessing": "string",  // e.g., "normalization", "one-hot encoding", "imputation"
                "rationale": "string"
            }}
        ]
        
        ONLY return the JSON array, no additional text.
        """
        
        response = self.query(prompt)
        
        try:
            # Try to parse as JSON
            suggestions = json.loads(response)
            return suggestions
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            # Return an empty list
            return []

class ExperimentTracker:
    """MLflow-based experiment tracking for ML model training."""
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "drift-ml-experiment"
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            tracking_uri: MLflow tracking URI
            experiment_name: Name of the experiment
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        self.experiment_name = experiment_name
        self._setup_experiment()
    
    def _setup_experiment(self) -> None:
        """Set up the MLflow experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(self.experiment_name)
            logger.info(f"Created new experiment: {self.experiment_name}")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {self.experiment_name}")
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.
        
        Args:
            params: Dictionary of parameters to log
        """
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                mlflow.log_param(key, json.dumps(value))
            else:
                mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log metrics to MLflow.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step for logging metrics (optional)
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str) -> None:
        """
        Log an artifact to MLflow.
        
        Args:
            local_path: Path to the artifact
        """
        mlflow.log_artifact(local_path)
    
    def log_model(self, model: Any, model_name: str, framework: str = "sklearn") -> None:
        """
        Log a model to MLflow.
        
        Args:
            model: Model to log
            model_name: Name for the model
            framework: Framework used for the model
        """
        if framework == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        elif framework == "pytorch":
            mlflow.pytorch.log_model(model, model_name)
        else:
            # Generic model logging using pickle
            with tempfile.TemporaryDirectory() as tmp_dir:
                model_path = os.path.join(tmp_dir, f"{model_name}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                self.log_artifact(model_path)
    
    def log_dataset(self, dataset: pd.DataFrame, name: str) -> None:
        """
        Log a dataset to MLflow.
        
        Args:
            dataset: DataFrame to log
            name: Name for the dataset
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset_path = os.path.join(tmp_dir, f"{name}.parquet")
            dataset.to_parquet(dataset_path, index=False)
            self.log_artifact(dataset_path)
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
        """
        mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
    
    def end_run(self) -> None:
        """End the current MLflow run."""
        mlflow.end_run()
    
    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """
        Get information about a run.
        
        Args:
            run_id: ID of the run
            
        Returns:
            Dictionary with run information
        """
        client = mlflow.tracking.MlflowClient()
        run = client.get_run(run_id)
        
        return {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "params": run.data.params,
            "metrics": run.data.metrics
        }
    
    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for runs in the experiment.
        
        Args:
            filter_string: Filter string for MLflow search
            max_results: Maximum number of results
            
        Returns:
            List of run information dictionaries
        """
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results
        )
        
        results = []
        for _, run in runs.iterrows():
            results.append({
                "run_id": run.run_id,
                "start_time": run.start_time,
                "metrics": {
                    col[len("metrics."):]: run[col]
                    for col in run.keys() if col.startswith("metrics.")
                },
                "params": {
                    col[len("params."):]: run[col]
                    for col in run.keys() if col.startswith("params.")
                }
            })
        
        return results
    
    def get_best_run(self, metric_name: str, ascending: bool = False) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric to optimize
            ascending: Whether to sort in ascending order
            
        Returns:
            Best run information or None if no runs found
        """
        order = "ASC" if ascending else "DESC"
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric_name} {order}"],
            max_results=1
        )
        
        if not runs.empty:
            run = runs.iloc[0]
            return {
                "run_id": run.run_id,
                "start_time": run.start_time,
                "metrics": {
                    col[len("metrics."):]: run[col]
                    for col in run.keys() if col.startswith("metrics.")
                },
                "params": {
                    col[len("params."):]: run[col]
                    for col in run.keys() if col.startswith("params.")
                }
            }
        
        return None

class ModelTrainer:
    """Base class for ML model training."""
    
    def __init__(
        self,
        experiment_tracker: Optional[ExperimentTracker] = None,
        llm_integration: Optional[LLMIntegration] = None
    ):
        """
        Initialize the model trainer.
        
        Args:
            experiment_tracker: Experiment tracker instance
            llm_integration: LLM integration instance
        """
        self.experiment_tracker = experiment_tracker
        self.llm_integration = llm_integration
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> Any:
        """
        Train a model (to be implemented by subclasses).
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a model (to be implemented by subclasses).
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test targets
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save a model to disk.
        
        Args:
            model: Model to save
            path: Path to save the model
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def load_model(self, path: str) -> Any:
        """
        Load a model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a model.
        
        Args:
            model: Model to use for predictions
            X: Features to predict
            
        Returns:
            Predictions
        """
        raise NotImplementedError("Subclasses must implement this method")
    
    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            data: Input DataFrame
            target_column: Name of the target column
            test_size: Fraction of data to use for testing
            val_size: Fraction of data to use for validation
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        # Split features and target
        X = data.drop(columns=[target_column]).values
        y = data[target_column].values
        
        # First split into train and test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Then split train into train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
        )
        
        return X_train, y_train, X_val, y_val, X_test, y_test

class SklearnModelTrainer(ModelTrainer):
    """Model trainer for scikit-learn models."""
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_type: str = "random_forest",
        hyperparams: Optional[Dict[str, Any]] = None,
        tune_hyperparams: bool = False,
        **kwargs
    ) -> Any:
        """
        Train a scikit-learn model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (not used for sklearn)
            y_val: Validation targets (not used for sklearn)
            model_type: Type of model to train
            hyperparams: Hyperparameters for the model
            tune_hyperparams: Whether to perform hyperparameter tuning
            **kwargs: Additional training parameters
            
        Returns:
            Trained model
        """
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        
        # Default hyperparameters
        default_hyperparams = {
            "random_forest": {"n_estimators": 100, "max_depth": 10},
            "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1},
            "logistic_regression": {"C": 1.0, "max_iter": 1000},
            "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale"}
        }
        
        # Get hyperparameters, using defaults if not provided
        if hyperparams is None:
            hyperparams = default_hyperparams.get(model_type, {})
        
        # Create model instance
        if model_type == "random_forest":
            model = RandomForestClassifier(**hyperparams, random_state=42)
        elif model_type == "gradient_boosting":
            model = GradientBoostingClassifier(**hyperparams, random_state=42)
        elif model_type == "logistic_regression":
            model = LogisticRegression(**hyperparams, random_state=42)
        elif model_type == "svm":
            model = SVC(**hyperparams, random_state=42, probability=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Track experiment if available
        if self.experiment_tracker:
            self.experiment_tracker.start_run(run_name=f"sklearn_{model_type}")
            
            # Log parameters
            params = {
                "model_type": model_type,
                "hyperparams": hyperparams,
                "X_train_shape": X_train.shape,
                "tune_hyperparams": tune_hyperparams
            }
            self.experiment_tracker.log_params(params)
        
        # Perform hyperparameter tuning if requested
        if tune_hyperparams:
            logger.info(f"Performing hyperparameter tuning for {model_type}")
            
            # Define param grid for tuning
            param_grid = {
                "random_forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [None, 10, 20, 30]
                },
                "gradient_boosting": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2]
                },
                "logistic_regression": {
                    "C": [0.1, 1.0, 10.0],
                    "solver": ["lbfgs", "liblinear"]
                },
                "svm": {
                    "C": [0.1, 1.0, 10.0],
                    "kernel": ["linear", "rbf", "poly"]
                }
            }
            
            grid = param_grid.get(model_type, {})
            
            if grid:
                grid_search = GridSearchCV(
                    model,
                    grid,
                    cv=5,
                    scoring="accuracy",
                    n_jobs=-1
                )
                
                grid_search.fit(X_train, y_train)
                
                # Update model with best parameters
                model = grid_search.best_estimator_
                
                if self.experiment_tracker:
                    self.experiment_tracker.log_params({"best_params": grid_search.best_params_})
            else:
                logger.warning(f"No parameter grid defined for {model_type}, skipping tuning")
        
        # Train the model
        logger.info(f"Training {model_type} model")
        model.fit(X_train, y_train)
        
        # Evaluate on training data
        train_preds = model.predict(X_train)
        train_metrics = self._compute_metrics(y_train, train_preds)
        
        logger.info(f"Training metrics: {train_metrics}")
        
        # Log metrics if tracking
        if self.experiment_tracker:
            train_metrics_prefixed = {f"train_{k}": v for k, v in train_metrics.items()}
            self.experiment_tracker.log_metrics(train_metrics_prefixed)
            
            # Log model
            self.experiment_tracker.log_model(model, f"sklearn_{model_type}", framework="sklearn")
            
            self.experiment_tracker.end_run()
        
        return model
    
    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a scikit-learn model.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test targets
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # For binary classification, also get probability predictions
        if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = None
        
        # Compute metrics
        metrics = self._compute_metrics(y_test, y_pred, y_proba)
        
        # Log metrics if tracking
        if self.experiment_tracker:
            self.experiment_tracker.start_run()
            self.experiment_tracker.log_metrics(metrics)
            self.experiment_tracker.end_run()
        
        return metrics
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred)
        }
        
        # For binary classification
        if len(np.unique(y_true)) == 2:
            metrics.update({
                "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="binary", zero_division=0)
            })
            
            # Add AUC if probabilities are available
            if y_proba is not None:
                metrics["auc"] = roc_auc_score(y_true, y_proba)
        
        # For multi-class classification
        elif len(np.unique(y_true)) > 2:
            metrics.update({
                "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)
            })
        
        return metrics
    
    def save_model(self, model: Any, path: str) -> None:
        """
        Save a scikit-learn model to disk.
        
        Args:
            model: Model to save
            path: Path to save the model
        """
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    def load_model(self, path: str) -> Any:
        """
        Load a scikit-learn model from disk.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Loaded model
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def predict(self, model: Any, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a scikit-learn model.
        
        Args:
            model: Model to use for predictions
            X: Features to predict
            
        Returns:
            Predictions
        """
        return model.predict(X)

class PyTorchModelTrainer(ModelTrainer):
    """Model trainer for PyTorch models."""
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        model_architecture: Optional[nn.Module] = None,
        hidden_layers: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 10,
        early_stopping_patience: Optional[int] = None,
        optimizer_type: str = "adam",
        loss_fn: Optional[Any] = None,
        **kwargs
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Train a PyTorch model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            model_architecture: Custom model architecture (optional)
            hidden_layers: List of hidden layer sizes (if model_architecture not provided)
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            early_stopping_patience: Number of epochs without improvement to stop
            optimizer_type: Type of optimizer to use
            loss_fn: Loss function to use
            **kwargs: Additional training parameters
            
        Returns:
            Tuple of (trained model, training history)
        """
        # Convert numpy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
            has_validation = True
        else:
            has_validation = False
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if has_validation:
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Determine input and output dimensions
        input_dim = X_train.shape[1]
        if len(np.unique(y_train)) == 2:
            output_dim = 1  # Binary classification
            task_type = "binary"
        else:
            output_dim = len(np.unique(y_train))  # Multi-class classification
            task_type = "multiclass"
        
        # Create model if not provided
        if model_architecture is None:
            if hidden_layers is None:
                hidden_layers = [128, 64]  # Default architecture
            
            model = self._create_mlp(
                input_dim=input_dim,
                hidden_layers=hidden_layers,
                output_dim=output_dim
            )
        else:
            model = model_architecture
        
        # Track experiment if available
        if self.experiment_tracker:
            self.experiment_tracker.start_run(run_name="pytorch_model")
            
            # Log parameters
            params = {
                "model_type": "pytorch",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": optimizer_type,
                "hidden_layers": hidden_layers,
                "early_stopping_patience": early_stopping_patience
            }
            self.experiment_tracker.log_params(params)
        
        # Define loss function
        if loss_fn is None:
            if task_type == "binary":
                loss_fn = nn.BCEWithLogitsLoss()
            else:
                loss_fn = nn.CrossEntropyLoss()
        
        # Define optimizer
        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": []
        }
        
        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        epochs_without_improvement = 0
        
        # Training loop
        logger.info("Starting PyTorch model training")
        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                # Forward pass
                if task_type == "binary":
                    targets = targets.float().unsqueeze(1)
                
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)
            
            # Validation
            if has_validation:
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        if task_type == "binary":
                            targets_loss = targets.float().unsqueeze(1)
                        else:
                            targets_loss = targets
                        
                        outputs = model(inputs)
                        loss = loss_fn(outputs, targets_loss)
                        val_loss += loss.item()
                        
                        # Calculate accuracy
                        if task_type == "binary":
                            predicted = (torch.sigmoid(outputs) > 0.5).int().squeeze()
                        else:
                            _, predicted = torch.max(outputs, 1)
                        
                        total += targets.size(0)
                        correct += (predicted == targets).sum().item()
                
                # Calculate average validation loss and accuracy
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct / total
                
                history["val_loss"].append(avg_val_loss)
                history["val_accuracy"].append(val_accuracy)
                
                # Log metrics if tracking
                if self.experiment_tracker:
                    metrics = {
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "val_accuracy": val_accuracy
                    }
                    self.experiment_tracker.log_metrics(metrics, step=epoch)
                
                # Early stopping
                if early_stopping_patience is not None:
                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        best_model_state = model.state_dict().copy()
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                    
                    if epochs_without_improvement >= early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                        break
                
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
            else:
                logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}")
        
        # Load best model if early stopping was used
        if early_stopping_patience is not None and best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Log model if tracking
        if self.experiment_tracker:
            self.experiment_tracker.log_model(model, "pytorch_model", framework="pytorch")
            self.experiment_tracker.end_run()
        
        return model, history
    
    def _create_mlp(self, input_dim: int, hidden_layers: List[int], output_dim: int) -> nn.Module:
        """
        Create a multi-layer perceptron (MLP) neural network.
        
        Args:
            input_dim: Input dimension
            hidden_layers: List of hidden layer sizes
            output_dim: Output dimension
            
        Returns:
            MLP neural network
        """
        # Define model architecture
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        
        return nn.Sequential(*layers)
    
    def evaluate(
        self,
        model: nn.Module,
        X_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate a PyTorch model.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test targets
            batch_size: Batch size
            **kwargs: Additional evaluation parameters
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Convert numpy arrays to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create data loader
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Determine if binary or multi-class classification
        if len(np.unique(y_test)) == 2:
            task_type = "binary"
        else:
            task_type = "multiclass"
        
        # Evaluation
        model.eval()
        all_predicted = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, _ in test_loader:
                outputs = model(inputs)
                
                # Get predictions
                if task_type == "binary":
                    probabilities = torch.sigmoid(outputs).squeeze().numpy()
                    predictions = (probabilities > 0.5).astype(int)
                else:
                    probabilities = torch.softmax(outputs, dim=1).numpy()
                    predictions = torch.argmax(outputs, dim=1).numpy()
                
                all_predicted.extend(predictions)
                
                # Store probabilities for ROC-AUC calculation
                if task_type == "binary":
                    all_probabilities.extend(probabilities)
                else:
                    all_probabilities.extend(probabilities)
        
        # Convert lists to numpy arrays
        all_predicted = np.array(all_predicted)
        all_probabilities = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self._compute_metrics(y_test, all_predicted, all_probabilities if task_type == "binary" else None)
        
        # Log metrics if tracking
        if self.experiment_tracker:
            self.experiment_tracker.start_run()
            self.experiment_tracker.log_metrics(metrics)
            self.experiment_tracker.end_run()
        
        return metrics
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred)
        }
        
        # For binary classification
        if len(np.unique(y_true)) == 2:
            metrics.update({
                "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
                "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
                "f1": f1_score(y_true, y_pred, average="binary", zero_division=0)
            })
            
            # Add AUC if probabilities are available
            if y_proba is not None:
                metrics["auc"] = roc_auc_score(y_true, y_proba)
        
        # For multi-class classification
        elif len(np.unique(y_true)) > 2:
            metrics.update({
                "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
                "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
                "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)
            })
        
        return metrics
    
    def save_model(self, model: nn.Module, path: str) -> None:
        """
        Save a PyTorch model to disk.
        
        Args:
            model: Model to save
            path: Path to save the model
        """
        torch.save(model.state_dict(), path)
    
    def load_model(self, path: str, model_architecture: nn.Module) -> nn.Module:
        """
        Load a PyTorch model from disk.
        
        Args:
            path: Path to load the model from
            model_architecture: Model architecture to load weights into
            
        Returns:
            Loaded model
        """
        model_architecture.load_state_dict(torch.load(path))
        return model_architecture
    
    def predict(self, model: nn.Module, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with a PyTorch model.
        
        Args:
            model: Model to use for predictions
            X: Features to predict
            
        Returns:
            Predictions
        """
        # Convert input to tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)
        
        # Set model to evaluation mode
        model.eval()
        
        # Make predictions
        with torch.no_grad():
            outputs = model(X_tensor)
            
            # Get predictions based on model output shape
            if outputs.shape[1] == 1:  # Binary classification
                predictions = (torch.sigmoid(outputs) > 0.5).int().squeeze().numpy()
            else:  # Multi-class classification
                predictions = torch.argmax(outputs, dim=1).numpy()
        
        return predictions

class ModelTrainerFactory:
    """Factory for creating model trainers."""
    
    @staticmethod
    def create_trainer(
        framework: str,
        experiment_tracker: Optional[ExperimentTracker] = None,
        llm_integration: Optional[LLMIntegration] = None
    ) -> ModelTrainer:
        """
        Create a model trainer for the specified framework.
        
        Args:
            framework: ML framework to use ("sklearn" or "pytorch")
            experiment_tracker: Experiment tracker instance
            llm_integration: LLM integration instance
            
        Returns:
            Model trainer instance
        """
        if framework.lower() == "sklearn":
            return SklearnModelTrainer(experiment_tracker, llm_integration)
        elif framework.lower() == "pytorch":
            return PyTorchModelTrainer(experiment_tracker, llm_integration)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

# Example usage
if __name__ == "__main__":
    # Setup experiment tracking and LLM integration
    tracker = ExperimentTracker(experiment_name="drift-ml-demo")
    llm = LLMIntegration(mode="offline", model_name="llama2")
    
    # Create a trainer for scikit-learn models
    sklearn_trainer = ModelTrainerFactory.create_trainer(
        framework="sklearn",
        experiment_tracker=tracker,
        llm_integration=llm
    )
    
    # Load and prepare data
    from sklearn.datasets import load_iris
    iris = load_iris()
    data = pd.DataFrame(
        data=np.c_[iris['data'], iris['target']],
        columns=[*iris['feature_names'], 'target']
    )
    
    # Prepare data for training
    X_train, y_train, X_val, y_val, X_test, y_test = sklearn_trainer.prepare_data(
        data=data,
        target_column='target'
    )
    
    # Get model suggestions from LLM
    dataset_description = "Iris dataset with 4 numerical features and 3 classes"
    suggestions = llm.suggest_model(dataset_description, task_type="classification")
    
    # Train a model
    model = sklearn_trainer.train(
        X_train=X_train,
        y_train=y_train,
        model_type="random_forest",
        tune_hyperparams=True
    )
    
    # Evaluate the model
    metrics = sklearn_trainer.evaluate(model, X_test, y_test)
    print(f"Model evaluation metrics: {metrics}")
    
    # Get explanation of results from LLM
    explanation = llm.explain_results(metrics, model_type="random_forest")
    print(f"LLM explanation: {explanation}")
    
    # Save the model
    sklearn_trainer.save_model(model, "iris_model.pkl")