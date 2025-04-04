"""
Automated model approval workflow for Drift-ML platform.
Evaluates model performance and approves or rejects models based on criteria.
"""
import os
import time
import json
from typing import Dict, List, Optional, Any, Union, Callable
import pandas as pd
import numpy as np
from loguru import logger

from src.model_registry.model_registry import ModelRegistry

class ApprovalCriteria:
    """Base class for model approval criteria."""
    
    def check(self, model_metrics: Dict[str, Any]) -> bool:
        """
        Check if model meets approval criteria.
        
        Args:
            model_metrics: Dictionary of model metrics
            
        Returns:
            True if model meets criteria, False otherwise
        """
        raise NotImplementedError("Subclasses must implement this method")

class MetricThresholdCriteria(ApprovalCriteria):
    """Approval criteria based on metric thresholds."""
    
    def __init__(
        self, 
        metric_name: str, 
        threshold: float, 
        comparison: str = ">=",
        required: bool = True
    ):
        """
        Initialize metric threshold criteria.
        
        Args:
            metric_name: Name of the metric to check
            threshold: Threshold value
            comparison: Comparison operator (>=, >, ==, <, <=)
            required: Whether the metric is required
        """
        self.metric_name = metric_name
        self.threshold = threshold
        self.comparison = comparison
        self.required = required
    
    def check(self, model_metrics: Dict[str, Any]) -> bool:
        """Check if metric meets threshold."""
        if self.metric_name not in model_metrics:
            return not self.required
        
        value = model_metrics[self.metric_name]
        
        if self.comparison == ">=":
            return value >= self.threshold
        elif self.comparison == ">":
            return value > self.threshold
        elif self.comparison == "==":
            return value == self.threshold
        elif self.comparison == "<":
            return value < self.threshold
        elif self.comparison == "<=":
            return value <= self.threshold
        else:
            logger.error(f"Unknown comparison operator: {self.comparison}")
            return False

class ComparisonCriteria(ApprovalCriteria):
    """Approval criteria based on comparison with another model."""
    
    def __init__(
        self,
        metric_name: str,
        minimum_improvement: float = 0.0,
        relative: bool = True,
        required: bool = False
    ):
        """
        Initialize comparison criteria.
        
        Args:
            metric_name: Name of the metric to compare
            minimum_improvement: Minimum improvement required
            relative: Whether improvement is relative (percentage) or absolute
            required: Whether the comparison is required
        """
        self.metric_name = metric_name
        self.minimum_improvement = minimum_improvement
        self.relative = relative
        self.required = required
    
    def check(self, current_metrics: Dict[str, Any], baseline_metrics: Dict[str, Any]) -> bool:
        """Check if current metrics show improvement over baseline."""
        if self.metric_name not in current_metrics or self.metric_name not in baseline_metrics:
            return not self.required
        
        current_value = current_metrics[self.metric_name]
        baseline_value = baseline_metrics[self.metric_name]
        
        if self.relative:
            # Avoid division by zero
            if baseline_value == 0:
                improvement = float('inf') if current_value > 0 else 0
            else:
                improvement = (current_value - baseline_value) / baseline_value
        else:
            improvement = current_value - baseline_value
        
        return improvement >= self.minimum_improvement

class AutomatedApprovalWorkflow:
    """Automated model approval workflow."""
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        approval_criteria: List[ApprovalCriteria],
        comparison_criteria: Optional[List[ComparisonCriteria]] = None,
        baseline_model_stage: str = "Production",
        approval_user: str = "automated-approval"
    ):
        """
        Initialize automated approval workflow.
        
        Args:
            model_registry: Model registry instance
            approval_criteria: List of approval criteria
            comparison_criteria: List of comparison criteria
            baseline_model_stage: Stage to use for baseline model
            approval_user: User to use for approval actions
        """
        self.model_registry = model_registry
        self.approval_criteria = approval_criteria
        self.comparison_criteria = comparison_criteria or []
        self.baseline_model_stage = baseline_model_stage
        self.approval_user = approval_user
    
    def evaluate_model(
        self,
        model_name: str,
        version: str,
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model against approval criteria.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            custom_metrics: Custom metrics to include in evaluation
            
        Returns:
            Dictionary with evaluation results
        """
        try:
            # Get model metrics
            metrics = self.model_registry.get_model_metrics(model_name, version)
            
            # Combine with custom metrics if provided
            if custom_metrics:
                metrics.update(custom_metrics)
            
            # Check if all approval criteria are met
            criteria_results = {}
            all_criteria_met = True
            
            for idx, criterion in enumerate(self.approval_criteria):
                criterion_name = getattr(criterion, 'metric_name', f"criterion_{idx}")
                result = criterion.check(metrics)
                criteria_results[criterion_name] = result
                
                if not result:
                    all_criteria_met = False
            
            # Get baseline model if comparison criteria exist
            comparison_results = {}
            
            if self.comparison_criteria:
                # Find baseline model in production stage
                baseline_versions = self.model_registry.client.get_latest_versions(
                    name=model_name,
                    stages=[self.baseline_model_stage]
                )
                
                if baseline_versions:
                    baseline_version = baseline_versions[0].version
                    
                    # Skip comparison if evaluating the baseline version
                    if baseline_version == version:
                        logger.info(f"Skipping comparison for {model_name} v{version} as it is the baseline")
                    else:
                        # Get baseline metrics
                        baseline_metrics = self.model_registry.get_model_metrics(model_name, baseline_version)
                        
                        for idx, criterion in enumerate(self.comparison_criteria):
                            criterion_name = f"compare_{getattr(criterion, 'metric_name', f'criterion_{idx}')}"
                            result = criterion.check(metrics, baseline_metrics)
                            comparison_results[criterion_name] = result
                            
                            if not result:
                                all_criteria_met = False
            
            return {
                "model_name": model_name,
                "model_version": version,
                "metrics": metrics,
                "criteria_results": criteria_results,
                "comparison_results": comparison_results,
                "all_criteria_met": all_criteria_met
            }
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name} v{version}: {e}")
            return {
                "model_name": model_name,
                "model_version": version,
                "error": str(e),
                "all_criteria_met": False
            }
    
    def run_approval_workflow(
        self,
        model_name: str,
        version: str,
        auto_transition: bool = False,
        transition_stage: str = "Staging",
        custom_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run approval workflow for a model version.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            auto_transition: Whether to automatically transition the model stage
            transition_stage: Stage to transition to if approved
            custom_metrics: Custom metrics to include in evaluation
            
        Returns:
            Dictionary with workflow results
        """
        try:
            # First check if model exists
            model_version = self.model_registry.get_model_version(model_name, version)
            if not model_version:
                return {
                    "model_name": model_name,
                    "model_version": version,
                    "status": "error",
                    "message": f"Model version {version} not found for model '{model_name}'"
                }
            
            # Check current approval status
            approval_status = self.model_registry.get_approval_status(model_name, version)
            if approval_status.get("approved"):
                return {
                    "model_name": model_name,
                    "model_version": version,
                    "status": "skipped",
                    "message": f"Model version already approved by {approval_status.get('approved_by')}"
                }
            
            if approval_status.get("rejected"):
                return {
                    "model_name": model_name,
                    "model_version": version,
                    "status": "skipped",
                    "message": f"Model version already rejected by {approval_status.get('rejected_by')}"
                }
            
            # Evaluate model
            evaluation = self.evaluate_model(model_name, version, custom_metrics)
            
            if evaluation.get("all_criteria_met", False):
                # Approve model
                self.model_registry.approve_model_version(
                    model_name=model_name,
                    version=version,
                    approver=self.approval_user,
                    comment="Automatically approved: all criteria met"
                )
                
                # Transition stage if requested
                if auto_transition:
                    self.model_registry.transition_model_version_stage(
                        model_name=model_name,
                        version=version,
                        stage=transition_stage
                    )
                
                return {
                    "model_name": model_name,
                    "model_version": version,
                    "status": "approved",
                    "message": "Model automatically approved: all criteria met",
                    "evaluation": evaluation,
                    "transitioned": auto_transition,
                    "stage": transition_stage if auto_transition else model_version.current_stage
                }
            else:
                # Reject model
                self.model_registry.reject_model_version(
                    model_name=model_name,
                    version=version,
                    rejector=self.approval_user,
                    reason="Automatically rejected: not all criteria met"
                )
                
                return {
                    "model_name": model_name,
                    "model_version": version,
                    "status": "rejected",
                    "message": "Model automatically rejected: not all criteria met",
                    "evaluation": evaluation
                }
                
        except Exception as e:
            logger.error(f"Error in approval workflow for {model_name} v{version}: {e}")
            return {
                "model_name": model_name,
                "model_version": version,
                "status": "error",
                "message": f"Error in approval workflow: {str(e)}"
            }

# Example usage
if __name__ == "__main__":
    registry = ModelRegistry(
        tracking_uri="http://localhost:5000",
    )
    
    # Define approval criteria
    criteria = [
        MetricThresholdCriteria("accuracy", 0.8, ">="),
        MetricThresholdCriteria("f1_score", 0.7, ">="),
        MetricThresholdCriteria("training_time", 300, "<=", required=False)
    ]
    
    # Define comparison criteria
    comparison = [
        ComparisonCriteria("accuracy", 0.01, relative=False),
        ComparisonCriteria("inference_latency_ms", -0.1, relative=True, required=False)
    ]
    
    workflow = AutomatedApprovalWorkflow(
        model_registry=registry,
        approval_criteria=criteria,
        comparison_criteria=comparison
    )
    
    # Run approval workflow
    result = workflow.run_approval_workflow(
        model_name="example_model",
        version="1",
        auto_transition=True,
        transition_stage="Staging"
    )
    
    print(json.dumps(result, indent=2))
