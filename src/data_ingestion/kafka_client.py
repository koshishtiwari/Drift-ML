"""
Data Ingestion module for Drift-ML platform.
Provides classes for integrating with Apache Kafka.
"""
import json
import uuid
from typing import Any, Dict, List, Optional

from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
from loguru import logger


class KafkaClient:
    """Base class for Kafka producer and consumer implementations."""
    
    def __init__(
        self, 
        bootstrap_servers: str,
        client_id: Optional[str] = None
    ):
        """
        Initialize the Kafka client.
        
        Args:
            bootstrap_servers: Comma-separated list of broker addresses
            client_id: Unique identifier for this client
        """
        self.bootstrap_servers = bootstrap_servers
        self.client_id = client_id or f"drift-ml-{uuid.uuid4()}"
        self.config = {
            'bootstrap.servers': bootstrap_servers,
            'client.id': self.client_id
        }
        
    def admin_client(self) -> AdminClient:
        """Create and return a Kafka AdminClient."""
        return AdminClient(self.config)
    
    def create_topics(self, topics: List[str], num_partitions: int = 1, replication_factor: int = 1) -> None:
        """
        Create Kafka topics if they don't exist.
        
        Args:
            topics: List of topic names to create
            num_partitions: Number of partitions for each topic
            replication_factor: Replication factor for each topic
        """
        admin = self.admin_client()
        new_topics = [
            NewTopic(
                topic, 
                num_partitions=num_partitions,
                replication_factor=replication_factor
            ) for topic in topics
        ]
        
        fs = admin.create_topics(new_topics)
        
        # Wait for operation to complete
        for topic, f in fs.items():
            try:
                f.result()  # The result itself is None
                logger.info(f"Topic {topic} created")
            except Exception as e:
                if "already exists" in str(e):
                    logger.info(f"Topic {topic} already exists")
                else:
                    logger.error(f"Failed to create topic {topic}: {e}")


class KafkaProducer(KafkaClient):
    """Kafka producer for sending data to Kafka topics."""
    
    def __init__(
        self, 
        bootstrap_servers: str,
        client_id: Optional[str] = None,
        acks: str = "all",
        **kwargs
    ):
        """
        Initialize the Kafka producer.
        
        Args:
            bootstrap_servers: Comma-separated list of broker addresses
            client_id: Unique identifier for this client
            acks: The number of acknowledgments the producer requires
            **kwargs: Additional producer configuration options
        """
        super().__init__(bootstrap_servers, client_id)
        self.config.update({
            'acks': acks,
        })
        self.config.update(kwargs)
        self.producer = Producer(self.config)
        
    def produce(self, topic: str, key: Optional[str], value: Dict[str, Any]) -> None:
        """
        Produce a message to a Kafka topic.
        
        Args:
            topic: Topic name
            key: Message key (optional)
            value: Message value as a dictionary
        """
        try:
            # Convert value to JSON string
            value_str = json.dumps(value)
            
            # Send message
            self.producer.produce(
                topic=topic,
                key=key.encode('utf-8') if key else None,
                value=value_str.encode('utf-8')
            )
            
            # Poll to handle delivery reports
            self.producer.poll(0)
            
        except Exception as e:
            logger.error(f"Error producing message to {topic}: {e}")
            raise
    
    def flush(self, timeout: float = 10.0) -> None:
        """
        Wait for all messages to be delivered.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        self.producer.flush(timeout)


class KafkaConsumer(KafkaClient):
    """Kafka consumer for consuming data from Kafka topics."""
    
    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: List[str],
        client_id: Optional[str] = None,
        auto_offset_reset: str = "earliest",
        **kwargs
    ):
        """
        Initialize the Kafka consumer.
        
        Args:
            bootstrap_servers: Comma-separated list of broker addresses
            group_id: Consumer group ID
            topics: List of topics to subscribe to
            client_id: Unique identifier for this client
            auto_offset_reset: Where to start reading ("earliest" or "latest")
            **kwargs: Additional consumer configuration options
        """
        super().__init__(bootstrap_servers, client_id)
        self.config.update({
            'group.id': group_id,
            'auto.offset.reset': auto_offset_reset,
        })
        self.config.update(kwargs)
        self.consumer = Consumer(self.config)
        self.topics = topics
        self.consumer.subscribe(topics)
        
    def consume(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """
        Consume a single message from subscribed topics.
        
        Args:
            timeout: Maximum time to wait for a message in seconds
            
        Returns:
            Dictionary containing message information or None if no message
        """
        msg = self.consumer.poll(timeout=timeout)
        
        if msg is None:
            return None
            
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                # End of partition event, not an error
                logger.debug(f"Reached end of partition {msg.partition()}")
                return None
            else:
                logger.error(f"Error consuming message: {msg.error()}")
                return None
                
        # Process valid message
        try:
            # Decode message value
            value_str = msg.value().decode('utf-8')
            value = json.loads(value_str)
            
            # Create result
            result = {
                'topic': msg.topic(),
                'partition': msg.partition(),
                'offset': msg.offset(),
                'key': msg.key().decode('utf-8') if msg.key() else None,
                'value': value,
                'timestamp': msg.timestamp()[1]
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return None
    
    def consume_batch(self, max_records: int = 100, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """
        Consume a batch of messages from subscribed topics.
        
        Args:
            max_records: Maximum number of records to consume
            timeout: Maximum time to wait for each message in seconds
            
        Returns:
            List of dictionaries containing message information
        """
        messages = []
        for _ in range(max_records):
            msg = self.consume(timeout=timeout)
            if msg is None:
                break
            messages.append(msg)
        return messages
    
    def close(self) -> None:
        """Close the consumer connection."""
        self.consumer.close()


class KafkaDataIngestion:
    """Main data ingestion class for Drift-ML platform using Kafka."""
    
    def __init__(
        self,
        bootstrap_servers: str,
        input_topics: List[str],
        output_topic: str,
        error_topic: str,
        group_id: str,
    ):
        """
        Initialize the data ingestion component.
        
        Args:
            bootstrap_servers: Kafka bootstrap servers
            input_topics: List of input topics to consume from
            output_topic: Topic to produce processed data to
            error_topic: Topic for error messages
            group_id: Consumer group ID
        """
        self.bootstrap_servers = bootstrap_servers
        self.input_topics = input_topics
        self.output_topic = output_topic
        self.error_topic = error_topic
        self.group_id = group_id
        
        # Create Kafka client and ensure topics exist
        self.client = KafkaClient(bootstrap_servers)
        topics_to_create = input_topics + [output_topic, error_topic]
        self.client.create_topics(topics_to_create)
        
        # Create producer and consumer
        self.producer = KafkaProducer(bootstrap_servers)
        self.consumer = KafkaConsumer(
            bootstrap_servers=bootstrap_servers,
            group_id=group_id,
            topics=input_topics
        )
        
    def process_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message (override in subclasses).
        
        Args:
            message: The message to process
            
        Returns:
            Processed message
        """
        # Default implementation passes the message through
        return message
    
    def run(self, batch_size: int = 100, timeout: float = 1.0) -> None:
        """
        Run the data ingestion process.
        
        Args:
            batch_size: Number of messages to process in each batch
            timeout: Timeout for consuming messages
        """
        try:
            logger.info(f"Starting data ingestion from topics: {self.input_topics}")
            
            while True:
                # Consume batch of messages
                messages = self.consumer.consume_batch(
                    max_records=batch_size,
                    timeout=timeout
                )
                
                if not messages:
                    continue
                
                logger.info(f"Consumed {len(messages)} messages")
                
                # Process each message
                for msg in messages:
                    try:
                        # Process message
                        processed_msg = self.process_message(msg)
                        
                        # Send to output topic
                        self.producer.produce(
                            topic=self.output_topic,
                            key=msg.get('key'),
                            value=processed_msg.get('value', {})
                        )
                        
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
                        # Send error to error topic
                        error_msg = {
                            'original_message': msg,
                            'error': str(e)
                        }
                        self.producer.produce(
                            topic=self.error_topic,
                            key=msg.get('key'),
                            value=error_msg
                        )
                
                # Ensure all messages are delivered
                self.producer.flush()
                
        except KeyboardInterrupt:
            logger.info("Shutting down data ingestion")
        finally:
            self.consumer.close()
            logger.info("Data ingestion stopped")