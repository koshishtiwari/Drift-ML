"""
Data Processing module for Drift-ML platform.
Provides classes for processing streaming data using Apache Flink.
"""
from typing import Dict, List, Any, Callable, Optional
import json
import os
from datetime import datetime

from pyflink.datastream import StreamExecutionEnvironment, TimeCharacteristic
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer
from pyflink.common.serialization import SimpleStringSchema
from pyflink.common.typeinfo import Types
from pyflink.datastream.functions import MapFunction, FilterFunction, KeyedProcessFunction
from loguru import logger

# Import security module
from src.security.security import Security


class JsonDeserializationSchema:
    """
    Deserialization schema for JSON data from Kafka.
    """
    def deserialize(self, kafka_message: bytes) -> Dict[str, Any]:
        """
        Deserialize Kafka message to a Python dictionary.
        
        Args:
            kafka_message: Raw message bytes from Kafka
            
        Returns:
            Deserialized message as a dictionary
        """
        try:
            message_str = kafka_message.decode('utf-8')
            return json.loads(message_str)
        except Exception as e:
            logger.error(f"Failed to deserialize message: {e}")
            return {'error': str(e), 'raw_message': kafka_message.decode('utf-8', errors='replace')}


class JsonSerializationSchema:
    """
    Serialization schema for JSON data to Kafka.
    """
    def serialize(self, element: Dict[str, Any]) -> bytes:
        """
        Serialize Python dictionary to bytes for Kafka.
        
        Args:
            element: Dictionary to serialize
            
        Returns:
            Serialized message as bytes
        """
        try:
            return json.dumps(element).encode('utf-8')
        except Exception as e:
            logger.error(f"Failed to serialize message: {e}")
            error_msg = {'error': str(e), 'timestamp': datetime.now().isoformat()}
            return json.dumps(error_msg).encode('utf-8')


class DataTransformation(MapFunction):
    """
    Base class for data transformations in Flink.
    """
    def map(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform a single record.
        
        Args:
            value: Input record
            
        Returns:
            Transformed record
        """
        # Default implementation passes through the data
        return value


class DataFilter(FilterFunction):
    """
    Base class for data filtering in Flink.
    """
    def filter(self, value: Dict[str, Any]) -> bool:
        """
        Filter a single record.
        
        Args:
            value: Input record
            
        Returns:
            True if the record should be kept, False otherwise
        """
        # Default implementation keeps all records
        return True


class FlinkProcessor:
    """
    Main data processing class for Drift-ML platform using Apache Flink.
    """
    def __init__(
        self,
        bootstrap_servers: str,
        input_topics: List[str],
        output_topic: str,
        error_topic: str,
        group_id: str,
        job_name: str = "drift-ml-data-processor"
    ):
        """
        Initialize the Flink data processor.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            input_topics: List of input topics to consume from
            output_topic: Topic to produce processed data to
            error_topic: Topic for error messages
            group_id: Consumer group ID
            job_name: Name of the Flink job
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topics = input_topics
        self.output_topic = output_topic
        self.error_topic = error_topic
        self.group_id = group_id
        self.job_name = job_name
        
        # Initialize Flink execution environment
        self.env = StreamExecutionEnvironment.get_execution_environment()
        self.env.set_stream_time_characteristic(TimeCharacteristic.EventTime)
        self.env.set_parallelism(1)  # Can be adjusted based on needs
        
        # Add Flink dependencies
        self._add_jars()
        
        # Transformation functions to apply (can be set by users)
        self.transformations: List[Callable] = []
        self.filters: List[Callable] = []
    
    def _add_jars(self) -> None:
        """Add required JAR files to the Flink environment."""
        jars = [
            "flink-connector-kafka_2.12",
            "flink-json",
            "kafka-clients"
        ]
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        jar_dir = os.path.join(current_dir, "..", "..", "lib")
        
        # If we have downloaded JARs locally
        if os.path.exists(jar_dir):
            for jar_file in os.listdir(jar_dir):
                if jar_file.endswith(".jar") and any(jar in jar_file for jar in jars):
                    self.env.add_jars(f"file://{os.path.join(jar_dir, jar_file)}")
    
    def add_transformation(self, transformation: Callable) -> None:
        """
        Add a transformation function to the pipeline.
        
        Args:
            transformation: Function that transforms data
        """
        self.transformations.append(transformation)
    
    def add_filter(self, filter_func: Callable) -> None:
        """
        Add a filter function to the pipeline.
        
        Args:
            filter_func: Function that filters data
        """
        self.filters.append(filter_func)
    
    def create_kafka_consumer(self, topic: str) -> FlinkKafkaConsumer:
        """
        Create a Kafka consumer for a specific topic.
        
        Args:
            topic: Topic name
            
        Returns:
            FlinkKafkaConsumer for the topic
        """
        props = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest'
        }
        
        return FlinkKafkaConsumer(
            topic,
            JsonDeserializationSchema(),
            properties=props
        )
    
    def create_kafka_producer(self, topic: str) -> FlinkKafkaProducer:
        """
        Create a Kafka producer for a specific topic.
        
        Args:
            topic: Topic name
            
        Returns:
            FlinkKafkaProducer for the topic
        """
        props = {
            'bootstrap.servers': self.bootstrap_servers,
            'transaction.timeout.ms': '5000'
        }
        
        return FlinkKafkaProducer(
            topic,
            JsonSerializationSchema(),
            properties=props
        )
    
    def _configure_kafka_source(self) -> None:
        """Configure Kafka source with backpressure handling."""
        properties = {
            'bootstrap.servers': self.kafka_bootstrap_servers,
            'group.id': f'{self.job_name}_group',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': 'false',
            # Backpressure handling settings
            'max.poll.records': '500',  # Limit records per poll
            'max.poll.interval.ms': '300000',  # 5 minutes
            'fetch.max.wait.ms': '500',  # Wait time for fetch
            'heartbeat.interval.ms': '3000',  # Heartbeat to broker
        }
        
        # Max in-flight requests per connection
        if self.high_throughput_mode:
            properties['max.in.flight.requests.per.connection'] = '5'
        else:
            properties['max.in.flight.requests.per.connection'] = '1'
        
        self.kafka_props = properties
        
        logger.info(f"Configured Kafka source with backpressure handling")

    def build_pipeline(self) -> None:
        """Build the Flink data processing pipeline with backpressure management."""
        # Create streams for each input topic
        streams = []
        for topic in self.input_topics:
            source = self.env.add_source(
                self.create_kafka_consumer(topic)
            ).name(f"Source: {topic}")
            streams.append(source)
        
        # Merge streams if we have multiple input topics
        if len(streams) > 1:
            stream = streams[0].union(*streams[1:])
        else:
            stream = streams[0]
        
        # Apply filters
        for filter_idx, filter_func in enumerate(self.filters):
            filter_name = getattr(filter_func, '__name__', f"Filter-{filter_idx}")
            if isinstance(filter_func, FilterFunction):
                stream = stream.filter(filter_func).name(filter_name)
            else:
                # If it's a regular function, wrap it
                stream = stream.filter(lambda x: filter_func(x)).name(filter_name)
        
        # Apply transformations
        for transform_idx, transform in enumerate(self.transformations):
            transform_name = getattr(transform, '__name__', f"Transform-{transform_idx}")
            if isinstance(transform, MapFunction):
                stream = stream.map(transform).name(transform_name)
            else:
                # If it's a regular function, wrap it
                stream = stream.map(lambda x: transform(x)).name(transform_name)
        
        # Add sink to the output topic
        stream.add_sink(
            self.create_kafka_producer(self.output_topic)
        ).name(f"Sink: {self.output_topic}")
        
        # Configure checkpointing for failure recovery
        self.env.enable_checkpointing(60000)  # 60 seconds
        self.env.get_checkpoint_config().set_min_pause_between_checkpoints(30000)  # 30 seconds
        self.env.get_checkpoint_config().set_checkpoint_timeout(20000)  # 20 seconds
        self.env.get_checkpoint_config().set_max_concurrent_checkpoints(1)
        self.env.get_checkpoint_config().set_tolerable_checkpoint_failure_number(3)
        
        # Configure restart strategy
        self.env.get_restart_strategy_configuration().set_restart_strategy_type_from_string("fixed-delay")
        self.env.get_restart_strategy_configuration().set_restart_attempts(3)
        self.env.get_restart_strategy_configuration().set_delay_between_attempts_interval_ms(10000)
        
        # Set up buffering and latency monitoring for backpressure detection
        self.env.get_configuration().set_string("pipeline.buffersperview.enabled", "true")
        self.env.get_configuration().set_string("pipeline.backpressure-monitoring.enabled", "true")
        
        # Add output monitoring to detect and log backpressure
        stream = stream.map(lambda x: self._monitor_backpressure(x))
        
    def _monitor_backpressure(self, record):
        """Monitor and log backpressure situations."""
        current_time = time.time()
        if hasattr(self, 'last_backpressure_check_time'):
            if current_time - self.last_backpressure_check_time > 60:  # Check every minute
                self.last_backpressure_check_time = current_time
                
                # Report processing metrics that can indicate backpressure
                if hasattr(self, 'records_processed'):
                    rate = self.records_processed / 60.0  # records per second
                    logger.info(f"Current processing rate: {rate:.2f} records/sec")
                    
                    # Reset counter
                    self.records_processed = 0
                    
                    # If we have previous metrics, check for backpressure
                    if hasattr(self, 'previous_rate'):
                        if self.previous_rate > 0 and rate / self.previous_rate < 0.7:
                            logger.warning(f"Possible backpressure detected: rate dropped from {self.previous_rate:.2f} to {rate:.2f}")
                    
                    self.previous_rate = rate
        else:
            self.last_backpressure_check_time = current_time
            self.records_processed = 0
            self.previous_rate = 0
        
        self.records_processed = getattr(self, 'records_processed', 0) + 1
        return record
    
    def run(self) -> None:
        """Run the Flink job."""
        try:
            logger.info(f"Starting Flink job: {self.job_name}")
            self.build_pipeline()
            self.env.execute(self.job_name)
        except Exception as e:
            logger.error(f"Error running Flink job: {e}")
            raise


class WindowedAggregator:
    """
    Helper class for windowed aggregations in Flink.
    """
    @staticmethod
    def create_tumbling_window(stream, window_size_ms: int, aggregation_function: Callable):
        """
        Create a tumbling window aggregation.
        
        Args:
            stream: The input DataStream
            window_size_ms: Window size in milliseconds
            aggregation_function: Function to apply to the window
            
        Returns:
            Windowed DataStream
        """
        return (
            stream
            .key_by(lambda x: x.get('key', 'default'))
            .window(TumblingEventTimeWindows.of(Time.milliseconds(window_size_ms)))
            .apply(aggregation_function)
        )
    
    @staticmethod
    def create_sliding_window(stream, window_size_ms: int, slide_size_ms: int, aggregation_function: Callable):
        """
        Create a sliding window aggregation.
        
        Args:
            stream: The input DataStream
            window_size_ms: Window size in milliseconds
            slide_size_ms: Slide size in milliseconds
            aggregation_function: Function to apply to the window
            
        Returns:
            Windowed DataStream
        """
        return (
            stream
            .key_by(lambda x: x.get('key', 'default'))
            .window(SlidingEventTimeWindows.of(
                Time.milliseconds(window_size_ms),
                Time.milliseconds(slide_size_ms)
            ))
            .apply(aggregation_function)
        )


class FeatureComputation(MapFunction):
    """
    Base class for feature computation in Flink.
    """
    def __init__(self, feature_name: str):
        """
        Initialize with a feature name.
        
        Args:
            feature_name: Name of the feature to compute
        """
        self.feature_name = feature_name
    
    def map(self, value: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute a feature from input data.
        
        Args:
            value: Input record
            
        Returns:
            Record with computed feature
        """
        # Implementation should be provided in subclasses
        result = value.copy()
        result[self.feature_name] = self._compute(value)
        return result
    
    def _compute(self, value: Dict[str, Any]) -> Any:
        """
        Compute the feature value (to be implemented by subclasses).
        
        Args:
            value: Input record
            
        Returns:
            Computed feature value
        """
        raise NotImplementedError("Subclasses must implement this method")


class SecureFlinkProcessor:
    """
    Secure data processing class with security integration.
    """
    
    def __init__(
        self,
        flink_processor: FlinkProcessor,
        security: Security
    ):
        """
        Initialize the secure Flink processor.
        
        Args:
            flink_processor: Base Flink processor implementation
            security: Security module
        """
        self.processor = flink_processor
        self.security = security
        self.job_id = None
    
    def _check_permission(self, user_id: int, action: str) -> bool:
        """
        Check if a user has permission to perform an action.
        
        Args:
            user_id: User ID
            action: Action to perform ("process", "view", etc.)
            
        Returns:
            True if permitted, False otherwise
        """
        try:
            return self.security.authz.check_permission(
                user_id=user_id,
                resource="data",
                action=action
            )
        except Exception as e:
            logger.error(f"Permission check failed: {e}")
            return False
    
    def _log_action(
        self, 
        action: str, 
        details: Optional[Dict[str, Any]],
        user_id: int,
        username: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> None:
        """
        Log an action in the audit log.
        
        Args:
            action: Action performed
            details: Additional details about the action
            user_id: User ID
            username: Username
            ip_address: IP address
        """
        try:
            self.security.audit.log_event(
                action=action,
                resource="data",
                resource_id=self.processor.job_name,
                user_id=user_id,
                username=username,
                details=details,
                ip_address=ip_address
            )
        except Exception as e:
            logger.error(f"Failed to log action: {e}")
    
    def add_transformation(
        self, 
        transformation: Callable,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Add a transformation function to the pipeline with security checks.
        
        Args:
            transformation: Function that transforms data
            user_id: User ID performing the action
            username: Username performing the action
            ip_address: IP address of the user
            
        Returns:
            True if successful, False if not authorized
        """
        if user_id is not None:
            # Check permission
            if not self._check_permission(user_id, "process"):
                logger.warning(f"User {user_id} does not have permission to process data")
                return False
            
            # Log the action
            transform_name = getattr(transformation, '__name__', "unnamed_transform")
            self._log_action(
                action="add_transformation",
                details={
                    "transformation_name": transform_name,
                    "job_name": self.processor.job_name
                },
                user_id=user_id,
                username=username,
                ip_address=ip_address
            )
        
        # Add the transformation
        self.processor.add_transformation(transformation)
        return True
    
    def add_filter(
        self, 
        filter_func: Callable,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Add a filter function to the pipeline with security checks.
        
        Args:
            filter_func: Function that filters data
            user_id: User ID performing the action
            username: Username performing the action
            ip_address: IP address of the user
            
        Returns:
            True if successful, False if not authorized
        """
        if user_id is not None:
            # Check permission
            if not self._check_permission(user_id, "process"):
                logger.warning(f"User {user_id} does not have permission to process data")
                return False
            
            # Log the action
            filter_name = getattr(filter_func, '__name__', "unnamed_filter")
            self._log_action(
                action="add_filter",
                details={
                    "filter_name": filter_name,
                    "job_name": self.processor.job_name
                },
                user_id=user_id,
                username=username,
                ip_address=ip_address
            )
        
        # Add the filter
        self.processor.add_filter(filter_func)
        return True
    
    def build_pipeline(
        self,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Build the Flink data processing pipeline with security checks.
        
        Args:
            user_id: User ID performing the action
            username: Username performing the action
            ip_address: IP address of the user
            
        Returns:
            True if successful, False if not authorized
        """
        if user_id is not None:
            # Check permission
            if not self._check_permission(user_id, "process"):
                logger.warning(f"User {user_id} does not have permission to process data")
                return False
            
            # Log the action
            self._log_action(
                action="build_pipeline",
                details={
                    "job_name": self.processor.job_name,
                    "input_topics": self.processor.input_topics,
                    "output_topic": self.processor.output_topic,
                    "num_transformations": len(self.processor.transformations),
                    "num_filters": len(self.processor.filters)
                },
                user_id=user_id,
                username=username,
                ip_address=ip_address
            )
        
        # Build the pipeline
        self.processor.build_pipeline()
        return True
    
    def run(
        self,
        user_id: Optional[int] = None,
        username: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Run the Flink job with security checks.
        
        Args:
            user_id: User ID performing the action
            username: Username performing the action
            ip_address: IP address of the user
            
        Returns:
            True if successful, False if not authorized or if an error occurs
        """
        if user_id is not None:
            # Check permission
            if not self._check_permission(user_id, "process"):
                logger.warning(f"User {user_id} does not have permission to process data")
                return False
            
            # Log the action
            self._log_action(
                action="start_job",
                details={
                    "job_name": self.processor.job_name,
                    "bootstrap_servers": self.processor.bootstrap_servers,
                    "input_topics": self.processor.input_topics,
                    "output_topic": self.processor.output_topic
                },
                user_id=user_id,
                username=username,
                ip_address=ip_address
            )
        
        # Run the job
        try:
            self.processor.run()
            
            # Log success if we have user ID
            if user_id is not None:
                self._log_action(
                    action="job_started",
                    details={
                        "job_name": self.processor.job_name,
                        "status": "success"
                    },
                    user_id=user_id,
                    username=username,
                    ip_address=ip_address
                )
            
            return True
        except Exception as e:
            logger.error(f"Error running Flink job: {e}")
            
            # Log failure if we have user ID
            if user_id is not None:
                self._log_action(
                    action="job_error",
                    details={
                        "job_name": self.processor.job_name,
                        "error": str(e)
                    },
                    user_id=user_id,
                    username=username,
                    ip_address=ip_address
                )
            
            return False

# Example usage with security integration
if __name__ == "__main__":
    # Initialize security
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
    
    # Create a basic Flink processor
    flink_processor = FlinkProcessor(
        bootstrap_servers="localhost:9092",
        input_topics=["input-topic"],
        output_topic="output-topic",
        error_topic="error-topic",
        group_id="drift-ml-group",
        job_name="secure-data-processing-job"
    )
    
    # Create secure Flink processor with security integration
    secure_processor = SecureFlinkProcessor(flink_processor, security)
    
    # Authenticate admin user
    auth_info = security.auth.authenticate_user(
        username="admin",
        password="secure-password"
    )
    
    if auth_info:
        # Define a simple transformation
        class AddTimestamp(MapFunction):
            def map(self, value: Dict[str, Any]) -> Dict[str, Any]:
                value['processing_timestamp'] = datetime.now().isoformat()
                return value
        
        # Add transformation with security
        secure_processor.add_transformation(
            AddTimestamp(),
            user_id=auth_info["id"],
            username=auth_info["username"],
            ip_address="127.0.0.1"
        )
        
        # Build the pipeline with security
        secure_processor.build_pipeline(
            user_id=auth_info["id"],
            username=auth_info["username"],
            ip_address="127.0.0.1"
        )
        
        # In a real application, you would run the job:
        # secure_processor.run(
        #     user_id=auth_info["id"],
        #     username=auth_info["username"],
        #     ip_address="127.0.0.1"
        # )
    else:
        print("Authentication failed")