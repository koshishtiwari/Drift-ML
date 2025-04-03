#!/usr/bin/env python
"""
Initialize and test security components for Drift-ML.
This script sets up the security database and tests the integration
with feature store and data processing components.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import security and components
from src.security.security import Security
from src.feature_store.feature_store import FeatureStore, FeatureValue, SecureFeatureStore
from src.data_processing.flink_processor import FlinkProcessor, SecureFlinkProcessor, MapFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_security() -> Security:
    """Initialize the security module."""
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

def test_feature_store_security(security: Security) -> None:
    """Test the security integration with the feature store."""
    logger.info("Testing feature store security integration...")
    
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
    
    try:
        # Create feature store
        feature_store = FeatureStore.from_config(config)
        secure_feature_store = SecureFeatureStore(feature_store, security)
        
        # Authenticate admin user
        auth_info = security.auth.authenticate_user(
            username=os.environ.get("ADMIN_USERNAME", "admin"),
            password=os.environ.get("ADMIN_PASSWORD", "secure-password")
        )
        
        if not auth_info:
            logger.error("Failed to authenticate admin user")
            return
        
        # Create feature metadata
        metadata = {
            "entity_type": "user",
            "value_type": "float",
            "description": "User average purchase amount in the last 30 days",
            "tags": ["user", "purchase", "monetary"],
            "statistics": {
                "min": 0.0,
                "max": 10000.0,
                "mean": 250.0
            }
        }
        
        # Save metadata with security
        result = secure_feature_store.save_metadata(
            feature_name="user_avg_purchase_30d",
            metadata=metadata,
            user_id=auth_info["id"],
            username=auth_info["username"],
            ip_address="127.0.0.1"
        )
        
        if result:
            logger.info("Successfully saved feature metadata with security")
        else:
            logger.error("Failed to save feature metadata")
        
        # Create a feature value
        feature = FeatureValue(
            entity_id="user123",
            feature_name="user_avg_purchase_30d",
            value=325.50,
            timestamp=datetime.utcnow(),
            metadata={"source": "transaction_db"}
        )
        
        # Save feature with security
        result = secure_feature_store.save_feature(
            feature=feature,
            user_id=auth_info["id"],
            username=auth_info["username"],
            ip_address="127.0.0.1"
        )
        
        if result:
            logger.info("Successfully saved feature value with security")
        else:
            logger.error("Failed to save feature value")
        
        # Get feature with security
        retrieved_feature = secure_feature_store.get_online_feature(
            entity_id="user123",
            feature_name="user_avg_purchase_30d",
            user_id=auth_info["id"],
            username=auth_info["username"],
            ip_address="127.0.0.1"
        )
        
        if retrieved_feature:
            logger.info(f"Successfully retrieved feature value: {retrieved_feature.value}")
        else:
            logger.error("Failed to retrieve feature value")
    
    except Exception as e:
        logger.error(f"Error testing feature store security: {e}")

def test_data_processing_security(security: Security) -> None:
    """Test the security integration with the data processing module."""
    logger.info("Testing data processing security integration...")
    
    # Create a basic Flink processor
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    try:
        flink_processor = FlinkProcessor(
            bootstrap_servers=bootstrap_servers,
            input_topics=["input-topic"],
            output_topic="output-topic",
            error_topic="error-topic",
            group_id="drift-ml-group",
            job_name="secure-data-processing-job"
        )
        
        # Create secure Flink processor
        secure_processor = SecureFlinkProcessor(flink_processor, security)
        
        # Authenticate admin user
        auth_info = security.auth.authenticate_user(
            username=os.environ.get("ADMIN_USERNAME", "admin"),
            password=os.environ.get("ADMIN_PASSWORD", "secure-password")
        )
        
        if not auth_info:
            logger.error("Failed to authenticate admin user")
            return
        
        # Define a simple transformation
        class AddTimestamp(MapFunction):
            def map(self, value: Dict[str, Any]) -> Dict[str, Any]:
                value['processing_timestamp'] = datetime.now().isoformat()
                return value
        
        # Add transformation with security
        result = secure_processor.add_transformation(
            AddTimestamp(),
            user_id=auth_info["id"],
            username=auth_info["username"],
            ip_address="127.0.0.1"
        )
        
        if result:
            logger.info("Successfully added transformation with security")
        else:
            logger.error("Failed to add transformation")
        
        # Build the pipeline with security
        result = secure_processor.build_pipeline(
            user_id=auth_info["id"],
            username=auth_info["username"],
            ip_address="127.0.0.1"
        )
        
        if result:
            logger.info("Successfully built pipeline with security")
        else:
            logger.error("Failed to build pipeline")
        
        logger.info("Data processing security integration test completed")
        
    except Exception as e:
        logger.error(f"Error testing data processing security: {e}")

def main() -> None:
    """Main function to initialize and test security components."""
    try:
        # Set up security
        security = setup_security()
        
        # Test feature store security integration
        test_feature_store_security(security)
        
        # Test data processing security integration
        test_data_processing_security(security)
        
        logger.info("Security integration tests completed successfully")
        
    except Exception as e:
        logger.error(f"Error during security initialization and testing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()