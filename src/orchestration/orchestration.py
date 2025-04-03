"""
Orchestration module for Drift-ML platform.
Provides workflow coordination and automation using Apache Airflow with LLM assistance.
"""
import os
import json
import time
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import httpx
import ollama
import google.generativeai as genai
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
from airflow.models import Variable, Connection
from airflow.hooks.base import BaseHook
from loguru import logger

class LLMAssistant:
    """LLM assistant for workflow guidance and natural language processing."""
    
    def __init__(
        self,
        mode: str = "offline",
        model_name: str = "llama2",
        api_key: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize the LLM assistant.
        
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
    
    def suggest_dag(self, task_description: str) -> Dict[str, Any]:
        """
        Generate a DAG suggestion based on task description.
        
        Args:
            task_description: Description of what the workflow should do
            
        Returns:
            Dictionary with DAG suggestion
        """
        prompt = f"""
        As an ML workflow expert, design an Apache Airflow DAG for the following task:
        
        {task_description}
        
        The response should be a valid JSON object with the following structure:
        
        {{
            "dag_id": "string",
            "default_args": {{
                "owner": "string",
                "retries": number,
                "retry_delay_minutes": number
            }},
            "schedule_interval": "string",
            "tasks": [
                {{
                    "task_id": "string",
                    "task_type": "string",  // python, bash, sensor, etc.
                    "dependencies": ["string"],  // task ids this task depends on
                    "parameters": {{}}  // task-specific parameters
                }}
            ]
        }}
        
        ONLY return the JSON object, no additional text.
        """
        
        response = self.query(prompt)
        
        try:
            # Try to parse as JSON
            dag_suggestion = json.loads(response)
            return dag_suggestion
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            # Return a default template
            return {
                "dag_id": "default_dag",
                "default_args": {
                    "owner": "drift-ml",
                    "retries": 1,
                    "retry_delay_minutes": 5
                },
                "schedule_interval": "0 0 * * *",
                "tasks": []
            }
    
    def explain_failure(self, task_id: str, error_logs: str) -> str:
        """
        Get LLM explanation for a task failure.
        
        Args:
            task_id: ID of the failed task
            error_logs: Error logs from the failed task
            
        Returns:
            Explanation and suggestion
        """
        prompt = f"""
        As an ML workflow debugging expert, explain the following error in task '{task_id}' and suggest a fix:
        
        {error_logs}
        
        Provide a concise explanation of what went wrong and a specific suggestion to fix it.
        """
        
        return self.query(prompt)
    
    def optimize_workflow(self, dag_json: Dict[str, Any], performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get suggestions for optimizing a workflow.
        
        Args:
            dag_json: Current DAG configuration
            performance_metrics: Performance metrics for the DAG
            
        Returns:
            Optimized DAG configuration
        """
        prompt = f"""
        As an ML workflow optimization expert, suggest improvements for the following workflow based on performance metrics:
        
        Current DAG:
        {json.dumps(dag_json, indent=2)}
        
        Performance Metrics:
        {json.dumps(performance_metrics, indent=2)}
        
        Return an optimized version of the DAG as a valid JSON object with the same structure.
        ONLY return the JSON object, no additional text.
        """
        
        response = self.query(prompt)
        
        try:
            # Try to parse as JSON
            optimized_dag = json.loads(response)
            return optimized_dag
        except json.JSONDecodeError:
            logger.error("Failed to parse LLM response as JSON")
            # Return the original DAG
            return dag_json

class WorkflowGenerator:
    """Generate Airflow workflows based on LLM suggestions."""
    
    def __init__(self, llm_assistant: LLMAssistant):
        """
        Initialize the workflow generator.
        
        Args:
            llm_assistant: LLM assistant instance
        """
        self.llm_assistant = llm_assistant
    
    def create_dag_from_description(self, task_description: str, output_path: str) -> None:
        """
        Create an Airflow DAG file from a natural language description.
        
        Args:
            task_description: Description of what the workflow should do
            output_path: Path to save the generated DAG file
        """
        # Get DAG suggestion from LLM
        dag_suggestion = self.llm_assistant.suggest_dag(task_description)
        
        # Generate Python code for the DAG
        dag_code = self._generate_dag_code(dag_suggestion)
        
        # Save to file
        with open(output_path, 'w') as f:
            f.write(dag_code)
        
        logger.info(f"Generated DAG file saved to {output_path}")
    
    def _generate_dag_code(self, dag_suggestion: Dict[str, Any]) -> str:
        """
        Generate Python code for an Airflow DAG.
        
        Args:
            dag_suggestion: DAG suggestion from LLM
            
        Returns:
            Python code for the DAG
        """
        dag_id = dag_suggestion.get("dag_id", "default_dag")
        default_args = dag_suggestion.get("default_args", {})
        schedule_interval = dag_suggestion.get("schedule_interval", "0 0 * * *")
        tasks = dag_suggestion.get("tasks", [])
        
        # Generate imports
        imports = [
            "from airflow import DAG",
            "from airflow.operators.python import PythonOperator",
            "from airflow.operators.bash import BashOperator",
            "from airflow.sensors.external_task import ExternalTaskSensor",
            "from airflow.utils.dates import days_ago",
            "from datetime import timedelta",
            "import os",
            "import sys",
            "",
            "# Add Drift-ML src directory to path",
            "sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))",
            "",
            "# Import Drift-ML components",
            "from src.feature_store.feature_store import FeatureStore",
            "from src.model_training.model_trainer import ModelTrainer",
            "from src.model_registry.model_registry import ModelRegistry",
            "from src.model_serving.model_server import ModelServer",
            ""
        ]
        
        # Generate default args
        default_args_code = [
            "default_args = {",
            f"    'owner': '{default_args.get('owner', 'drift-ml')}',",
            f"    'retries': {default_args.get('retries', 1)},",
            f"    'retry_delay': timedelta(minutes={default_args.get('retry_delay_minutes', 5)}),",
            f"    'start_date': days_ago(1),",
            "}"
        ]
        
        # Generate DAG instantiation
        dag_code = [
            "",
            f"dag = DAG(",
            f"    '{dag_id}',",
            f"    default_args=default_args,",
            f"    description='Generated DAG for {dag_id}',",
            f"    schedule_interval='{schedule_interval}',",
            f"    catchup=False",
            f")",
            ""
        ]
        
        # Generate task definitions
        task_codes = []
        for task in tasks:
            task_id = task.get("task_id", "task")
            task_type = task.get("task_type", "python")
            params = task.get("parameters", {})
            
            if task_type == "python":
                function_name = f"{task_id}_function"
                task_codes.append(f"def {function_name}(**context):")
                task_codes.append(f"    # TODO: Implement {task_id} logic")
                task_codes.append(f"    print('Executing {task_id}')")
                task_codes.append("")
                
                task_codes.append(f"{task_id} = PythonOperator(")
                task_codes.append(f"    task_id='{task_id}',")
                task_codes.append(f"    python_callable={function_name},")
                task_codes.append(f"    dag=dag")
                task_codes.append(")")
                task_codes.append("")
            
            elif task_type == "bash":
                bash_command = params.get("bash_command", "echo 'Hello'")
                task_codes.append(f"{task_id} = BashOperator(")
                task_codes.append(f"    task_id='{task_id}',")
                task_codes.append(f"    bash_command='{bash_command}',")
                task_codes.append(f"    dag=dag")
                task_codes.append(")")
                task_codes.append("")
            
            elif task_type == "sensor":
                external_dag_id = params.get("external_dag_id", "other_dag")
                external_task_id = params.get("external_task_id", "other_task")
                task_codes.append(f"{task_id} = ExternalTaskSensor(")
                task_codes.append(f"    task_id='{task_id}',")
                task_codes.append(f"    external_dag_id='{external_dag_id}',")
                task_codes.append(f"    external_task_id='{external_task_id}',")
                task_codes.append(f"    timeout=300,")
                task_codes.append(f"    mode='reschedule',")
                task_codes.append(f"    dag=dag")
                task_codes.append(")")
                task_codes.append("")
        
        # Generate task dependencies
        dependency_codes = []
        for task in tasks:
            task_id = task.get("task_id", "task")
            dependencies = task.get("dependencies", [])
            
            if dependencies:
                upstream_tasks = " >> ".join(dependencies)
                dependency_codes.append(f"[{upstream_tasks}] >> {task_id}")
        
        # Combine all code sections
        all_code = (
            imports +
            default_args_code +
            dag_code +
            task_codes +
            (["", "# Task dependencies"] if dependency_codes else []) +
            dependency_codes
        )
        
        return "\n".join(all_code)

class WorkflowOrchestrator:
    """Main orchestration class for Drift-ML platform."""
    
    def __init__(
        self,
        airflow_home: str,
        llm_mode: str = "offline",
        llm_model_name: str = "llama2",
        api_key: Optional[str] = None
    ):
        """
        Initialize the workflow orchestrator.
        
        Args:
            airflow_home: Airflow home directory
            llm_mode: LLM mode ("offline" or "online")
            llm_model_name: LLM model name
            api_key: API key for online LLM (if applicable)
        """
        self.airflow_home = airflow_home
        
        # Create LLM assistant
        self.llm_assistant = LLMAssistant(
            mode=llm_mode,
            model_name=llm_model_name,
            api_key=api_key
        )
        
        # Create workflow generator
        self.workflow_generator = WorkflowGenerator(self.llm_assistant)
    
    def create_workflow(self, name: str, description: str) -> None:
        """
        Create a new workflow from description.
        
        Args:
            name: Workflow name
            description: Natural language description of the workflow
        """
        # Create DAG file
        dags_dir = os.path.join(self.airflow_home, "dags")
        os.makedirs(dags_dir, exist_ok=True)
        
        output_path = os.path.join(dags_dir, f"{name}.py")
        self.workflow_generator.create_dag_from_description(description, output_path)
    
    def list_workflows(self) -> List[str]:
        """
        List available workflows.
        
        Returns:
            List of workflow names
        """
        dags_dir = os.path.join(self.airflow_home, "dags")
        
        if not os.path.exists(dags_dir):
            return []
        
        return [
            f.replace(".py", "") 
            for f in os.listdir(dags_dir) 
            if f.endswith(".py") and not f.startswith("_")
        ]
    
    def get_workflow_status(self, workflow_name: str) -> Dict[str, Any]:
        """
        Get status of a workflow using Airflow API.
        
        Args:
            workflow_name: Workflow name
            
        Returns:
            Workflow status information
        """
        # This is a simplified version - in a real implementation, 
        # use Airflow REST API or database queries
        try:
            # Example of using httpx to query Airflow API
            # In real implementation, get credentials from a secure source
            response = httpx.get(
                f"http://localhost:8080/api/v1/dags/{workflow_name}",
                auth=("airflow", "airflow")
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get workflow status: {response.text}")
                return {"error": f"Failed to get status: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return {"error": str(e)}
    
    def trigger_workflow(self, workflow_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Trigger a workflow run.
        
        Args:
            workflow_name: Workflow name
            config: Configuration for the workflow run
            
        Returns:
            True if workflow was triggered successfully, False otherwise
        """
        try:
            # Example of using httpx to trigger a DAG via Airflow API
            data = {"conf": config} if config else {}
            
            response = httpx.post(
                f"http://localhost:8080/api/v1/dags/{workflow_name}/dagRuns",
                json=data,
                auth=("airflow", "airflow")
            )
            
            if response.status_code in (200, 201):
                logger.info(f"Triggered workflow {workflow_name}")
                return True
            else:
                logger.error(f"Failed to trigger workflow: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error triggering workflow: {e}")
            return False
    
    def get_workflow_logs(self, workflow_name: str, run_id: str, task_id: str) -> str:
        """
        Get logs for a specific task in a workflow run.
        
        Args:
            workflow_name: Workflow name
            run_id: Run ID
            task_id: Task ID
            
        Returns:
            Task logs
        """
        try:
            # Example of using httpx to get task logs via Airflow API
            response = httpx.get(
                f"http://localhost:8080/api/v1/dags/{workflow_name}/dagRuns/{run_id}/taskInstances/{task_id}/logs",
                auth=("airflow", "airflow")
            )
            
            if response.status_code == 200:
                return response.text
            else:
                logger.error(f"Failed to get task logs: {response.text}")
                return f"Failed to get logs: {response.status_code}"
                
        except Exception as e:
            logger.error(f"Error getting task logs: {e}")
            return f"Error: {str(e)}"

# Example workflow templates for common ML tasks
class WorkflowTemplates:
    """Common workflow templates for ML tasks."""
    
    @staticmethod
    def feature_engineering_workflow() -> Dict[str, Any]:
        """Template for feature engineering workflow."""
        return {
            "dag_id": "feature_engineering",
            "default_args": {
                "owner": "drift-ml",
                "retries": 3,
                "retry_delay_minutes": 5
            },
            "schedule_interval": "0 */6 * * *",  # Every 6 hours
            "tasks": [
                {
                    "task_id": "check_data_sources",
                    "task_type": "python",
                    "dependencies": [],
                    "parameters": {}
                },
                {
                    "task_id": "extract_data",
                    "task_type": "python",
                    "dependencies": ["check_data_sources"],
                    "parameters": {}
                },
                {
                    "task_id": "compute_features",
                    "task_type": "python",
                    "dependencies": ["extract_data"],
                    "parameters": {}
                },
                {
                    "task_id": "store_features",
                    "task_type": "python",
                    "dependencies": ["compute_features"],
                    "parameters": {}
                },
                {
                    "task_id": "validate_features",
                    "task_type": "python",
                    "dependencies": ["store_features"],
                    "parameters": {}
                }
            ]
        }
    
    @staticmethod
    def model_training_workflow() -> Dict[str, Any]:
        """Template for model training workflow."""
        return {
            "dag_id": "model_training",
            "default_args": {
                "owner": "drift-ml",
                "retries": 2,
                "retry_delay_minutes": 10
            },
            "schedule_interval": "0 0 * * *",  # Daily at midnight
            "tasks": [
                {
                    "task_id": "prepare_training_data",
                    "task_type": "python",
                    "dependencies": [],
                    "parameters": {}
                },
                {
                    "task_id": "train_model",
                    "task_type": "python",
                    "dependencies": ["prepare_training_data"],
                    "parameters": {}
                },
                {
                    "task_id": "evaluate_model",
                    "task_type": "python",
                    "dependencies": ["train_model"],
                    "parameters": {}
                },
                {
                    "task_id": "register_model",
                    "task_type": "python",
                    "dependencies": ["evaluate_model"],
                    "parameters": {}
                }
            ]
        }
    
    @staticmethod
    def model_deployment_workflow() -> Dict[str, Any]:
        """Template for model deployment workflow."""
        return {
            "dag_id": "model_deployment",
            "default_args": {
                "owner": "drift-ml",
                "retries": 2,
                "retry_delay_minutes": 5
            },
            "schedule_interval": None,  # Triggered manually
            "tasks": [
                {
                    "task_id": "get_model_from_registry",
                    "task_type": "python",
                    "dependencies": [],
                    "parameters": {}
                },
                {
                    "task_id": "validate_model",
                    "task_type": "python",
                    "dependencies": ["get_model_from_registry"],
                    "parameters": {}
                },
                {
                    "task_id": "deploy_model",
                    "task_type": "python",
                    "dependencies": ["validate_model"],
                    "parameters": {}
                },
                {
                    "task_id": "run_smoke_tests",
                    "task_type": "python",
                    "dependencies": ["deploy_model"],
                    "parameters": {}
                },
                {
                    "task_id": "update_model_status",
                    "task_type": "python",
                    "dependencies": ["run_smoke_tests"],
                    "parameters": {}
                }
            ]
        }

# Example DAG for feature engineering workflow
def create_feature_engineering_dag():
    """
    Create a DAG for feature engineering workflow.
    
    Returns:
        Airflow DAG
    """
    # Define default arguments
    default_args = {
        'owner': 'drift-ml',
        'depends_on_past': False,
        'start_date': days_ago(1),
        'email_on_failure': False,
        'email_on_retry': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
    }
    
    # Create DAG
    dag = DAG(
        'feature_engineering',
        default_args=default_args,
        description='Feature engineering workflow',
        schedule_interval='0 */6 * * *',  # Every 6 hours
        catchup=False,
    )
    
    # Define tasks
    def check_data_sources(**kwargs):
        """Check that all required data sources are available."""
        # Example implementation
        logger.info("Checking data sources")
        return True
    
    def extract_data(**kwargs):
        """Extract data from sources."""
        # Example implementation
        logger.info("Extracting data")
        return {"data_path": "/tmp/extracted_data.csv"}
    
    def compute_features(**kwargs):
        """Compute features from extracted data."""
        # Get data path from upstream task
        ti = kwargs['ti']
        extract_result = ti.xcom_pull(task_ids='extract_data')
        data_path = extract_result.get('data_path')
        
        # Example implementation
        logger.info(f"Computing features from {data_path}")
        return {"features_path": "/tmp/computed_features.json"}
    
    def store_features(**kwargs):
        """Store features in the feature store."""
        # Get features path from upstream task
        ti = kwargs['ti']
        compute_result = ti.xcom_pull(task_ids='compute_features')
        features_path = compute_result.get('features_path')
        
        # Example implementation
        logger.info(f"Storing features from {features_path}")
        
        # In a real implementation, use the FeatureStore class
        # from src.feature_store.feature_store import FeatureStore
        # feature_store = FeatureStore.from_config(...)
        # feature_store.save_feature(...)
        
        return {"status": "success"}
    
    def validate_features(**kwargs):
        """Validate stored features."""
        # Example implementation
        logger.info("Validating features")
        return {"validation_status": "passed"}
    
    # Create task instances
    check_sources_task = PythonOperator(
        task_id='check_data_sources',
        python_callable=check_data_sources,
        dag=dag,
    )
    
    extract_data_task = PythonOperator(
        task_id='extract_data',
        python_callable=extract_data,
        dag=dag,
    )
    
    compute_features_task = PythonOperator(
        task_id='compute_features',
        python_callable=compute_features,
        dag=dag,
    )
    
    store_features_task = PythonOperator(
        task_id='store_features',
        python_callable=store_features,
        dag=dag,
    )
    
    validate_features_task = PythonOperator(
        task_id='validate_features',
        python_callable=validate_features,
        dag=dag,
    )
    
    # Define task dependencies
    check_sources_task >> extract_data_task >> compute_features_task >> store_features_task >> validate_features_task
    
    return dag

# Example usage
if __name__ == "__main__":
    # Initialize orchestrator
    orchestrator = WorkflowOrchestrator(
        airflow_home="/opt/airflow",
        llm_mode="offline",
        llm_model_name="llama2"
    )
    
    # Create a workflow from a natural language description
    description = """
    Create a workflow that:
    1. Extracts customer transaction data from Kafka
    2. Computes aggregated features (daily spend, average transaction size, etc.)
    3. Stores the features in the feature store
    4. Triggers model retraining if feature distributions change significantly
    """
    
    orchestrator.create_workflow("customer_features", description)
    
    # List available workflows
    workflows = orchestrator.list_workflows()
    print(f"Available workflows: {workflows}")
    
    # Get workflow status
    status = orchestrator.get_workflow_status("customer_features")
    print(f"Workflow status: {status}")
    
    # Trigger workflow
    success = orchestrator.trigger_workflow("customer_features")
    print(f"Workflow triggered: {success}")