"""
Main entry point for the Drift-ML platform.
This module initializes all components and starts the API server.
"""

import os
import sys
import logging
import argparse
from typing import Dict, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import security and components
from src.security.security import Security
from src.feature_store.feature_store import FeatureStore, SecureFeatureStore
from src.model_serving.model_server import LocalModelServer
from src.data_processing.flink_processor import FlinkProcessor, SecureFlinkProcessor

def initialize_security() -> Optional[Security]:
    """Initialize the security module."""
    try:
        # Get security config from environment or use defaults
        db_url = os.environ.get("SECURITY_DB_URL", "sqlite:///data/security.db")
        jwt_secret = os.environ.get("JWT_SECRET", "your-jwt-secret-key")
        token_expiry = int(os.environ.get("TOKEN_EXPIRY", "3600"))
        
        logger.info(f"Initializing security with database: {db_url}")
        
        # Create security instance
        security = Security(
            db_url=db_url,
            jwt_secret=jwt_secret,
            token_expiry=token_expiry
        )
        
        # Set up default roles and permissions
        security.setup_default_roles_and_permissions()
        logger.info("Default roles and permissions created")
        
        # Create admin user if it doesn't exist
        admin_username = os.environ.get("ADMIN_USERNAME", "admin")
        admin_email = os.environ.get("ADMIN_EMAIL", "admin@example.com")
        admin_password = os.environ.get("ADMIN_PASSWORD", "secure-password")
        
        # Authenticate to check if admin exists
        auth_info = security.auth.authenticate_user(
            username=admin_username,
            password=admin_password
        )
        
        if not auth_info:
            # Create admin if it doesn't exist
            admin_id = security.create_initial_admin_user(
                username=admin_username,
                email=admin_email,
                password=admin_password
            )
            logger.info(f"Admin user created with ID: {admin_id}")
        else:
            logger.info(f"Admin user already exists with ID: {auth_info['id']}")
        
        return security
    
    except Exception as e:
        logger.error(f"Failed to initialize security: {e}")
        return None

def initialize_feature_store(security: Optional[Security] = None) -> Optional[SecureFeatureStore]:
    """Initialize the feature store with security."""
    try:
        # Create a feature store configuration
        config = {
            "online_store": {
                "type": "redis",
                "host": os.environ.get("REDIS_HOST", "localhost"),
                "port": int(os.environ.get("REDIS_PORT", "6379")),
                "db": 0
            },
            "offline_store": {
                "type": "postgres",
                "host": os.environ.get("POSTGRES_HOST", "localhost"),
                "port": int(os.environ.get("POSTGRES_PORT", "5432")),
                "dbname": os.environ.get("POSTGRES_DB", "featurestore"),
                "user": os.environ.get("POSTGRES_USER", "driftml"),
                "password": os.environ.get("POSTGRES_PASSWORD", "driftml")
            }
        }
        
        # Create feature store
        feature_store = FeatureStore.from_config(config)
        
        # Wrap with security if provided
        if security:
            return SecureFeatureStore(feature_store, security)
        else:
            logger.warning("Security not provided, returning unsecured feature store")
            return None
    
    except Exception as e:
        logger.error(f"Failed to initialize feature store: {e}")
        return None

def initialize_model_server(security: Optional[Security] = None) -> Optional[LocalModelServer]:
    """Initialize the model server with security."""
    try:
        # Get configuration from environment
        host = os.environ.get("MODEL_SERVER_HOST", "0.0.0.0")
        port = int(os.environ.get("MODEL_SERVER_PORT", "8000"))
        model_registry_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlflow:5000")
        
        # Create model server with security if provided
        server = LocalModelServer(
            model_registry_uri=model_registry_uri,
            host=host,
            port=port,
            security=security
        )
        
        return server
    
    except Exception as e:
        logger.error(f"Failed to initialize model server: {e}")
        return None

def initialize_data_processor(security: Optional[Security] = None) -> Optional[SecureFlinkProcessor]:
    """Initialize the data processor with security."""
    try:
        # Get configuration from environment
        bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        input_topics = os.environ.get("KAFKA_INPUT_TOPICS", "input-topic").split(",")
        output_topic = os.environ.get("KAFKA_OUTPUT_TOPIC", "output-topic")
        error_topic = os.environ.get("KAFKA_ERROR_TOPIC", "error-topic")
        group_id = os.environ.get("KAFKA_GROUP_ID", "drift-ml-group")
        
        # Create Flink processor
        flink_processor = FlinkProcessor(
            bootstrap_servers=bootstrap_servers,
            input_topics=input_topics,
            output_topic=output_topic,
            error_topic=error_topic,
            group_id=group_id
        )
        
        # Wrap with security if provided
        if security:
            return SecureFlinkProcessor(flink_processor, security)
        else:
            logger.warning("Security not provided, returning unsecured data processor")
            return None
    
    except Exception as e:
        logger.error(f"Failed to initialize data processor: {e}")
        return None

def main() -> None:
    """Main entry point for the Drift-ML platform."""
    parser = argparse.ArgumentParser(description="Start Drift-ML platform")
    parser.add_argument("--no-security", action="store_true", help="Disable security features")
    args = parser.parse_args()
    
    try:
        # Initialize security unless disabled
        security = None if args.no_security else initialize_security()
        
        # Initialize components with security
        feature_store = initialize_feature_store(security)
        model_server = initialize_model_server(security)
        data_processor = initialize_data_processor(security)
        
        # Start the model server (this will block and serve the API)
        if model_server:
            logger.info(f"Starting model server on {model_server.host}:{model_server.port}")
            model_server.start()
        else:
            logger.error("Failed to start model server")
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"Error starting Drift-ML platform: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()