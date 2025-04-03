"""
Feature Store module for Drift-ML platform.
Provides functionality for feature computation, storage, and retrieval.
"""

import json
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import redis
import boto3
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, Column, String, Float, Integer, DateTime, JSON, MetaData, Table, select, and_, desc, func, inspect
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from loguru import logger

# Import security module
from src.security.security import Security

# Base class for SQLAlchemy models
Base = declarative_base()


class FeatureMetadata(Base):
    """SQLAlchemy model for feature metadata."""
    __tablename__ = "feature_metadata"
    
    feature_name = Column(String, primary_key=True)
    entity_type = Column(String, nullable=False)
    value_type = Column(String, nullable=False)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    tags = Column(JSON)
    statistics = Column(JSON)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary."""
        return {
            "feature_name": self.feature_name,
            "entity_type": self.entity_type,
            "value_type": self.value_type,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "tags": self.tags,
            "statistics": self.statistics
        }


class FeatureValue:
    """Model class for feature values."""
    def __init__(
        self,
        entity_id: str,
        feature_name: str,
        value: Any,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a feature value.
        
        Args:
            entity_id: Identifier for the entity
            feature_name: Name of the feature
            value: Feature value
            timestamp: Timestamp when the feature value was computed
            metadata: Additional metadata for the feature value
        """
        self.entity_id = entity_id
        self.feature_name = feature_name
        self.value = value
        self.timestamp = timestamp or datetime.utcnow()
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert feature value to dictionary."""
        return {
            "entity_id": self.entity_id,
            "feature_name": self.feature_name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureValue":
        """Create feature value from dictionary."""
        timestamp = data.get("timestamp")
        if timestamp and isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        return cls(
            entity_id=data["entity_id"],
            feature_name=data["feature_name"],
            value=data["value"],
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )


class RedisFeatureStore:
    """Online feature store implementation using Redis."""
    
    def __init__(self, host: str, port: int, db: int = 0, password: Optional[str] = None):
        """
        Initialize the Redis feature store.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
        """
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True
        )
        self._check_connection()
    
    def _check_connection(self) -> None:
        """Check if Redis connection is working."""
        try:
            self.client.ping()
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    def _get_key(self, entity_id: str, feature_name: str) -> str:
        """
        Generate Redis key for a feature value.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            
        Returns:
            Redis key
        """
        return f"feature:{feature_name}:{entity_id}"
    
    def _get_latest_key(self, entity_id: str, feature_name: str) -> str:
        """
        Generate Redis key for the latest feature value.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            
        Returns:
            Redis key for latest value
        """
        return f"latest:feature:{feature_name}:{entity_id}"
    
    def _get_metadata_key(self, feature_name: str) -> str:
        """
        Generate Redis key for feature metadata.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Redis key for metadata
        """
        return f"metadata:{feature_name}"
    
    def save_feature(self, feature: FeatureValue) -> None:
        """
        Save a feature value to Redis.
        
        Args:
            feature: Feature value to save
        """
        key = self._get_key(feature.entity_id, feature.feature_name)
        latest_key = self._get_latest_key(feature.entity_id, feature.feature_name)
        
        # Convert feature to JSON string
        feature_json = json.dumps(feature.to_dict())
        
        # Use transaction to ensure atomicity
        pipe = self.client.pipeline()
        try:
            # Store historical values in a sorted set with timestamp as score
            timestamp_ms = int(feature.timestamp.timestamp() * 1000)
            pipe.zadd(key, {feature_json: timestamp_ms})
            
            # Store latest value
            pipe.set(latest_key, feature_json)
            
            # Set expiration on historical values if needed
            # pipe.expire(key, 86400 * 30)  # 30 days TTL
            
            pipe.execute()
            logger.debug(f"Saved feature {feature.feature_name} for entity {feature.entity_id}")
        except redis.RedisError as e:
            logger.error(f"Failed to save feature: {e}")
            raise
    
    def get_feature(
        self, 
        entity_id: str, 
        feature_name: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[FeatureValue]:
        """
        Get a feature value from Redis.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            timestamp: Timestamp to get feature value at (if None, get latest)
            
        Returns:
            Feature value or None if not found
        """
        if timestamp is None:
            # Get latest value
            key = self._get_latest_key(entity_id, feature_name)
            try:
                value = self.client.get(key)
                if value is None:
                    return None
                return FeatureValue.from_dict(json.loads(value))
            except (redis.RedisError, json.JSONDecodeError) as e:
                logger.error(f"Failed to get latest feature value: {e}")
                return None
        else:
            # Get historical value at or before the timestamp
            key = self._get_key(entity_id, feature_name)
            timestamp_ms = int(timestamp.timestamp() * 1000)
            
            try:
                # Get values with scores less than or equal to timestamp_ms
                values = self.client.zrevrangebyscore(
                    key,
                    timestamp_ms,
                    float('-inf'),
                    start=0,
                    num=1
                )
                
                if not values:
                    return None
                
                return FeatureValue.from_dict(json.loads(values[0]))
            except (redis.RedisError, json.JSONDecodeError) as e:
                logger.error(f"Failed to get historical feature value: {e}")
                return None
    
    def get_features(
        self,
        entity_id: str,
        feature_names: List[str],
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Optional[FeatureValue]]:
        """
        Get multiple feature values for an entity.
        
        Args:
            entity_id: Entity ID
            feature_names: List of feature names
            timestamp: Timestamp to get feature values at (if None, get latest)
            
        Returns:
            Dictionary mapping feature names to feature values
        """
        result = {}
        
        for feature_name in feature_names:
            result[feature_name] = self.get_feature(
                entity_id=entity_id,
                feature_name=feature_name,
                timestamp=timestamp
            )
        
        return result
    
    def save_metadata(self, feature_name: str, metadata: Dict[str, Any]) -> None:
        """
        Save metadata for a feature.
        
        Args:
            feature_name: Feature name
            metadata: Metadata to save
        """
        key = self._get_metadata_key(feature_name)
        try:
            self.client.set(key, json.dumps(metadata))
            logger.debug(f"Saved metadata for feature {feature_name}")
        except redis.RedisError as e:
            logger.error(f"Failed to save metadata: {e}")
            raise
    
    def get_metadata(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Feature metadata or None if not found
        """
        key = self._get_metadata_key(feature_name)
        try:
            value = self.client.get(key)
            if value is None:
                return None
            return json.loads(value)
        except (redis.RedisError, json.JSONDecodeError) as e:
            logger.error(f"Failed to get metadata: {e}")
            return None


class PostgresFeatureStore:
    """Offline feature store implementation using PostgreSQL."""
    
    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str
    ):
        """
        Initialize the PostgreSQL feature store.
        
        Args:
            host: PostgreSQL host
            port: PostgreSQL port
            dbname: Database name
            user: Database user
            password: Database password
        """
        self.connection_string = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.connection_string)
        self.Session = sessionmaker(bind=self.engine)
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database tables."""
        try:
            # Create metadata table
            Base.metadata.create_all(self.engine)
            
            # Create connection
            with self.engine.connect() as conn:
                metadata = MetaData()
                
                # Check if feature values table exists
                inspector = inspect(self.engine)
                if "feature_values" not in inspector.get_table_names():
                    # Create feature values table
                    Table(
                        "feature_values",
                        metadata,
                        Column("entity_id", String, primary_key=True),
                        Column("feature_name", String, primary_key=True),
                        Column("timestamp", DateTime, primary_key=True),
                        Column("value_json", JSON, nullable=False),
                        Column("metadata_json", JSON),
                    )
                    metadata.create_all(self.engine)
                    logger.info("Created feature_values table")
            
            logger.info("Successfully initialized PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def save_feature(self, feature: FeatureValue) -> None:
        """
        Save a feature value to PostgreSQL.
        
        Args:
            feature: Feature value to save
        """
        try:
            with self.engine.connect() as conn:
                # Insert feature value
                query = """
                INSERT INTO feature_values 
                (entity_id, feature_name, timestamp, value_json, metadata_json)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (entity_id, feature_name, timestamp) 
                DO UPDATE SET 
                    value_json = EXCLUDED.value_json,
                    metadata_json = EXCLUDED.metadata_json
                """
                
                conn.execute(
                    query,
                    (
                        feature.entity_id,
                        feature.feature_name,
                        feature.timestamp,
                        json.dumps(feature.value),
                        json.dumps(feature.metadata)
                    )
                )
                
                logger.debug(f"Saved feature {feature.feature_name} for entity {feature.entity_id} to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to save feature to PostgreSQL: {e}")
            raise
    
    def save_features_batch(self, features: List[FeatureValue]) -> None:
        """
        Save multiple feature values in a batch.
        
        Args:
            features: List of feature values to save
        """
        if not features:
            return
        
        try:
            with self.engine.connect() as conn:
                # Prepare values for batch insert
                values = [
                    (
                        feature.entity_id,
                        feature.feature_name,
                        feature.timestamp,
                        json.dumps(feature.value),
                        json.dumps(feature.metadata)
                    )
                    for feature in features
                ]
                
                # Execute batch insert
                query = """
                INSERT INTO feature_values 
                (entity_id, feature_name, timestamp, value_json, metadata_json)
                VALUES %s
                ON CONFLICT (entity_id, feature_name, timestamp) 
                DO UPDATE SET 
                    value_json = EXCLUDED.value_json,
                    metadata_json = EXCLUDED.metadata_json
                """
                
                execute_values(conn, query, values)
                
                logger.debug(f"Saved {len(features)} features in batch to PostgreSQL")
        except Exception as e:
            logger.error(f"Failed to save features batch to PostgreSQL: {e}")
            raise
    
    def get_feature(
        self,
        entity_id: str,
        feature_name: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[FeatureValue]:
        """
        Get a feature value from PostgreSQL.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            timestamp: Timestamp to get feature value at (if None, get latest)
            
        Returns:
            Feature value or None if not found
        """
        try:
            with self.engine.connect() as conn:
                if timestamp is None:
                    # Get latest value
                    query = """
                    SELECT entity_id, feature_name, timestamp, value_json, metadata_json
                    FROM feature_values
                    WHERE entity_id = %s AND feature_name = %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """
                    result = conn.execute(query, (entity_id, feature_name)).fetchone()
                else:
                    # Get value at or before timestamp
                    query = """
                    SELECT entity_id, feature_name, timestamp, value_json, metadata_json
                    FROM feature_values
                    WHERE entity_id = %s AND feature_name = %s AND timestamp <= %s
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """
                    result = conn.execute(query, (entity_id, feature_name, timestamp)).fetchone()
                
                if result is None:
                    return None
                
                entity_id, feature_name, timestamp, value_json, metadata_json = result
                
                return FeatureValue(
                    entity_id=entity_id,
                    feature_name=feature_name,
                    value=json.loads(value_json),
                    timestamp=timestamp,
                    metadata=json.loads(metadata_json) if metadata_json else {}
                )
        except Exception as e:
            logger.error(f"Failed to get feature from PostgreSQL: {e}")
            return None
    
    def get_feature_history(
        self,
        entity_id: str,
        feature_name: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[FeatureValue]:
        """
        Get historical feature values.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            start_time: Start time for history
            end_time: End time for history
            limit: Maximum number of values to return
            
        Returns:
            List of historical feature values
        """
        try:
            with self.engine.connect() as conn:
                query_parts = [
                    "SELECT entity_id, feature_name, timestamp, value_json, metadata_json",
                    "FROM feature_values",
                    "WHERE entity_id = %s AND feature_name = %s"
                ]
                params = [entity_id, feature_name]
                
                if start_time:
                    query_parts.append("AND timestamp >= %s")
                    params.append(start_time)
                
                if end_time:
                    query_parts.append("AND timestamp <= %s")
                    params.append(end_time)
                
                query_parts.append("ORDER BY timestamp DESC")
                query_parts.append("LIMIT %s")
                params.append(limit)
                
                query = " ".join(query_parts)
                
                results = conn.execute(query, params).fetchall()
                
                return [
                    FeatureValue(
                        entity_id=row[0],
                        feature_name=row[1],
                        value=json.loads(row[3]),
                        timestamp=row[2],
                        metadata=json.loads(row[4]) if row[4] else {}
                    )
                    for row in results
                ]
        except Exception as e:
            logger.error(f"Failed to get feature history from PostgreSQL: {e}")
            return []
    
    def get_training_data(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        training_time: datetime
    ) -> pd.DataFrame:
        """
        Get training data with point-in-time correctness.
        
        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names
            training_time: Reference time for training
            
        Returns:
            DataFrame with training data
        """
        try:
            with self.engine.connect() as conn:
                # This query gets the latest feature value for each entity and feature
                # that is less than or equal to the training time
                query = """
                WITH latest_features AS (
                    SELECT 
                        entity_id,
                        feature_name,
                        value_json,
                        ROW_NUMBER() OVER (
                            PARTITION BY entity_id, feature_name 
                            ORDER BY timestamp DESC
                        ) as row_num
                    FROM feature_values
                    WHERE 
                        entity_id IN %s AND
                        feature_name IN %s AND
                        timestamp <= %s
                )
                SELECT entity_id, feature_name, value_json
                FROM latest_features
                WHERE row_num = 1
                """
                
                results = conn.execute(
                    query, 
                    (tuple(entity_ids), tuple(feature_names), training_time)
                ).fetchall()
                
                # Create a dictionary for DataFrame construction
                data = {
                    "entity_id": [],
                }
                
                # Initialize feature columns
                for feature_name in feature_names:
                    data[feature_name] = []
                
                # Group results by entity_id
                entity_features = {}
                for row in results:
                    entity_id, feature_name, value_json = row
                    
                    if entity_id not in entity_features:
                        entity_features[entity_id] = {}
                    
                    entity_features[entity_id][feature_name] = json.loads(value_json)
                
                # Fill the data dictionary
                for entity_id in entity_ids:
                    data["entity_id"].append(entity_id)
                    
                    for feature_name in feature_names:
                        if entity_id in entity_features and feature_name in entity_features[entity_id]:
                            data[feature_name].append(entity_features[entity_id][feature_name])
                        else:
                            data[feature_name].append(None)
                
                return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to get training data from PostgreSQL: {e}")
            return pd.DataFrame()
    
    def save_metadata(self, metadata: FeatureMetadata) -> None:
        """
        Save feature metadata.
        
        Args:
            metadata: Feature metadata
        """
        session = self.Session()
        try:
            session.merge(metadata)
            session.commit()
            logger.debug(f"Saved metadata for feature {metadata.feature_name}")
        except Exception as e:
            session.rollback()
            logger.error(f"Failed to save metadata: {e}")
            raise
        finally:
            session.close()
    
    def get_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """
        Get metadata for a feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Feature metadata or None if not found
        """
        session = self.Session()
        try:
            return session.query(FeatureMetadata).filter_by(feature_name=feature_name).first()
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return None
        finally:
            session.close()
    
    def list_features(self, entity_type: Optional[str] = None) -> List[FeatureMetadata]:
        """
        List available features.
        
        Args:
            entity_type: Filter by entity type
            
        Returns:
            List of feature metadata
        """
        session = self.Session()
        try:
            query = session.query(FeatureMetadata)
            
            if entity_type:
                query = query.filter_by(entity_type=entity_type)
            
            return query.all()
        except Exception as e:
            logger.error(f"Failed to list features: {e}")
            return []
        finally:
            session.close()


class S3FeatureStore:
    """Offline feature store implementation using S3-compatible storage."""
    
    def __init__(
        self,
        bucket_name: str,
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        region_name: Optional[str] = None
    ):
        """
        Initialize the S3 feature store.
        
        Args:
            bucket_name: S3 bucket name
            endpoint_url: S3 endpoint URL (for non-AWS S3 compatible storage)
            aws_access_key_id: AWS access key ID
            aws_secret_access_key: AWS secret access key
            region_name: AWS region name
        """
        self.bucket_name = bucket_name
        
        # Initialize S3 client
        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint_url,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
    
    def _ensure_bucket_exists(self) -> None:
        """Ensure the S3 bucket exists."""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            logger.info(f"S3 bucket {self.bucket_name} exists")
        except Exception:
            logger.info(f"Creating S3 bucket {self.bucket_name}")
            try:
                self.s3.create_bucket(Bucket=self.bucket_name)
            except Exception as e:
                logger.error(f"Failed to create S3 bucket: {e}")
                raise
    
    def _get_feature_key(self, entity_id: str, feature_name: str, timestamp: datetime) -> str:
        """
        Generate S3 key for a feature value.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            timestamp: Timestamp
            
        Returns:
            S3 key
        """
        timestamp_str = timestamp.strftime("%Y/%m/%d/%H/%M/%S")
        return f"features/{feature_name}/{entity_id}/{timestamp_str}.json"
    
    def _get_metadata_key(self, feature_name: str) -> str:
        """
        Generate S3 key for feature metadata.
        
        Args:
            feature_name: Feature name
            
        Returns:
            S3 key
        """
        return f"metadata/{feature_name}.json"
    
    def save_feature(self, feature: FeatureValue) -> None:
        """
        Save a feature value to S3.
        
        Args:
            feature: Feature value to save
        """
        key = self._get_feature_key(feature.entity_id, feature.feature_name, feature.timestamp)
        
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(feature.to_dict())
            )
            logger.debug(f"Saved feature {feature.feature_name} for entity {feature.entity_id} to S3")
        except Exception as e:
            logger.error(f"Failed to save feature to S3: {e}")
            raise
    
    def save_features_batch(self, features: List[FeatureValue]) -> None:
        """
        Save multiple feature values in a batch.
        
        Args:
            features: List of feature values to save
        """
        if not features:
            return
        
        for feature in features:
            try:
                self.save_feature(feature)
            except Exception as e:
                logger.error(f"Failed to save feature {feature.feature_name} for entity {feature.entity_id}: {e}")
    
    def get_feature(
        self,
        entity_id: str,
        feature_name: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[FeatureValue]:
        """
        Get a feature value from S3.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            timestamp: Timestamp to get feature value at (if None, get latest)
            
        Returns:
            Feature value or None if not found
        """
        if timestamp is None:
            # Find the latest feature value
            prefix = f"features/{feature_name}/{entity_id}/"
            try:
                response = self.s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix,
                    MaxKeys=1000
                )
                
                if "Contents" not in response:
                    return None
                
                # Sort by key (which includes timestamp) to find the latest
                objects = sorted(response["Contents"], key=lambda x: x["Key"], reverse=True)
                
                if not objects:
                    return None
                
                # Get the latest object
                latest_key = objects[0]["Key"]
                
                response = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key=latest_key
                )
                
                data = json.loads(response["Body"].read().decode("utf-8"))
                return FeatureValue.from_dict(data)
                
            except Exception as e:
                logger.error(f"Failed to get latest feature from S3: {e}")
                return None
        else:
            # This is an approximation - S3 doesn't have efficient time-based queries
            # For production, consider using a database for queries and S3 for storage
            prefix = f"features/{feature_name}/{entity_id}/"
            try:
                response = self.s3.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=prefix
                )
                
                if "Contents" not in response:
                    return None
                
                # Filter objects by timestamp
                valid_objects = []
                for obj in response["Contents"]:
                    key = obj["Key"]
                    parts = key.split("/")
                    
                    # Extract timestamp components from key
                    if len(parts) >= 7:
                        try:
                            obj_timestamp = datetime(
                                int(parts[-7]),  # year
                                int(parts[-6]),  # month
                                int(parts[-5]),  # day
                                int(parts[-4]),  # hour
                                int(parts[-3]),  # minute
                                int(parts[-2].split('.')[0])  # second
                            )
                            
                            if obj_timestamp <= timestamp:
                                valid_objects.append((key, obj_timestamp))
                        except (ValueError, IndexError):
                            continue
                
                if not valid_objects:
                    return None
                
                # Sort by timestamp to find the latest before the specified timestamp
                valid_objects.sort(key=lambda x: x[1], reverse=True)
                
                latest_key = valid_objects[0][0]
                
                response = self.s3.get_object(
                    Bucket=self.bucket_name,
                    Key=latest_key
                )
                
                data = json.loads(response["Body"].read().decode("utf-8"))
                return FeatureValue.from_dict(data)
                
            except Exception as e:
                logger.error(f"Failed to get feature at timestamp from S3: {e}")
                return None
    
    def save_metadata(self, feature_name: str, metadata: Dict[str, Any]) -> None:
        """
        Save metadata for a feature.
        
        Args:
            feature_name: Feature name
            metadata: Metadata to save
        """
        key = self._get_metadata_key(feature_name)
        
        try:
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=json.dumps(metadata)
            )
            logger.debug(f"Saved metadata for feature {feature_name} to S3")
        except Exception as e:
            logger.error(f"Failed to save metadata to S3: {e}")
            raise
    
    def get_metadata(self, feature_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a feature.
        
        Args:
            feature_name: Feature name
            
        Returns:
            Feature metadata or None if not found
        """
        key = self._get_metadata_key(feature_name)
        
        try:
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=key
            )
            
            return json.loads(response["Body"].read().decode("utf-8"))
        except self.s3.exceptions.NoSuchKey:
            return None
        except Exception as e:
            logger.error(f"Failed to get metadata from S3: {e}")
            return None


class FeatureStore:
    """Main feature store class that combines online and offline stores."""
    
    def __init__(
        self,
        online_store: Optional[RedisFeatureStore] = None,
        offline_store: Optional[Union[PostgresFeatureStore, S3FeatureStore]] = None
    ):
        """
        Initialize the feature store.
        
        Args:
            online_store: Online feature store (Redis)
            offline_store: Offline feature store (PostgreSQL or S3)
        """
        self.online_store = online_store
        self.offline_store = offline_store
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "FeatureStore":
        """
        Create a feature store from configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Feature store instance
        """
        online_store = None
        offline_store = None
        
        # Configure online store
        if "online_store" in config:
            online_config = config["online_store"]
            if online_config["type"] == "redis":
                online_store = RedisFeatureStore(
                    host=online_config.get("host", "localhost"),
                    port=online_config.get("port", 6379),
                    db=online_config.get("db", 0),
                    password=online_config.get("password")
                )
        
        # Configure offline store
        if "offline_store" in config:
            offline_config = config["offline_store"]
            if offline_config["type"] == "postgres":
                offline_store = PostgresFeatureStore(
                    host=offline_config.get("host", "localhost"),
                    port=offline_config.get("port", 5432),
                    dbname=offline_config.get("dbname", "featurestore"),
                    user=offline_config.get("user", "postgres"),
                    password=offline_config.get("password", "postgres")
                )
            elif offline_config["type"] == "s3":
                offline_store = S3FeatureStore(
                    bucket_name=offline_config.get("bucket_name", "featurestore"),
                    endpoint_url=offline_config.get("endpoint_url"),
                    aws_access_key_id=offline_config.get("aws_access_key_id"),
                    aws_secret_access_key=offline_config.get("aws_secret_access_key"),
                    region_name=offline_config.get("region_name")
                )
        
        return cls(online_store=online_store, offline_store=offline_store)
    
    def save_feature(
        self, 
        feature: FeatureValue,
        store_type: str = "both"
    ) -> None:
        """
        Save a feature value.
        
        Args:
            feature: Feature value to save
            store_type: Where to save ("online", "offline", or "both")
        """
        if store_type in ["online", "both"] and self.online_store:
            self.online_store.save_feature(feature)
        
        if store_type in ["offline", "both"] and self.offline_store:
            self.offline_store.save_feature(feature)
    
    def get_online_feature(
        self,
        entity_id: str,
        feature_name: str
    ) -> Optional[FeatureValue]:
        """
        Get a feature value from the online store.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            
        Returns:
            Feature value or None if not found or no online store
        """
        if self.online_store:
            return self.online_store.get_feature(entity_id, feature_name)
        return None
    
    def get_offline_feature(
        self,
        entity_id: str,
        feature_name: str,
        timestamp: Optional[datetime] = None
    ) -> Optional[FeatureValue]:
        """
        Get a feature value from the offline store.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            timestamp: Timestamp to get feature value at
            
        Returns:
            Feature value or None if not found or no offline store
        """
        if self.offline_store:
            return self.offline_store.get_feature(entity_id, feature_name, timestamp)
        return None
    
    def get_online_features(
        self,
        entity_id: str,
        feature_names: List[str]
    ) -> Dict[str, Optional[FeatureValue]]:
        """
        Get multiple feature values from the online store.
        
        Args:
            entity_id: Entity ID
            feature_names: List of feature names
            
        Returns:
            Dictionary mapping feature names to feature values
        """
        result = {feature_name: None for feature_name in feature_names}
        
        if self.online_store:
            return self.online_store.get_features(entity_id, feature_names)
        
        return result
    
    def get_training_data(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        training_time: datetime
    ) -> pd.DataFrame:
        """
        Get training data with point-in-time correctness.
        
        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names
            training_time: Reference time for training
            
        Returns:
            DataFrame with training data
        """
        if isinstance(self.offline_store, PostgresFeatureStore):
            return self.offline_store.get_training_data(
                entity_ids=entity_ids,
                feature_names=feature_names,
                training_time=training_time
            )
        
        # Fallback implementation for S3 or other offline stores
        # This is less efficient as it requires multiple queries
        data = {
            "entity_id": entity_ids
        }
        
        for feature_name in feature_names:
            data[feature_name] = []
            
            for entity_id in entity_ids:
                feature = self.get_offline_feature(
                    entity_id=entity_id,
                    feature_name=feature_name,
                    timestamp=training_time
                )
                
                data[feature_name].append(feature.value if feature else None)
        
        return pd.DataFrame(data)
    
    def save_metadata(
        self,
        feature_name: str,
        metadata: Dict[str, Any],
        store_type: str = "both"
    ) -> None:
        """
        Save metadata for a feature.
        
        Args:
            feature_name: Feature name
            metadata: Metadata to save
            store_type: Where to save ("online", "offline", or "both")
        """
        if store_type in ["online", "both"] and self.online_store:
            self.online_store.save_metadata(feature_name, metadata)
        
        if store_type in ["offline", "both"] and self.offline_store:
            if isinstance(self.offline_store, PostgresFeatureStore):
                # Convert to SQLAlchemy model
                feature_metadata = FeatureMetadata(
                    feature_name=feature_name,
                    entity_type=metadata.get("entity_type", "unknown"),
                    value_type=metadata.get("value_type", "unknown"),
                    description=metadata.get("description"),
                    tags=metadata.get("tags"),
                    statistics=metadata.get("statistics")
                )
                self.offline_store.save_metadata(feature_metadata)
            else:
                self.offline_store.save_metadata(feature_name, metadata)
    
    def get_metadata(
        self,
        feature_name: str,
        store_type: str = "offline"
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a feature.
        
        Args:
            feature_name: Feature name
            store_type: Where to get from ("online" or "offline")
            
        Returns:
            Feature metadata or None if not found
        """
        if store_type == "online" and self.online_store:
            return self.online_store.get_metadata(feature_name)
        
        if store_type == "offline" and self.offline_store:
            metadata = self.offline_store.get_metadata(feature_name)
            
            if isinstance(metadata, FeatureMetadata):
                return metadata.to_dict()
            
            return metadata
        
        return None
    
    def list_features(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List available features.
        
        Args:
            entity_type: Filter by entity type
            
        Returns:
            List of feature metadata
        """
        if isinstance(self.offline_store, PostgresFeatureStore):
            features = self.offline_store.list_features(entity_type)
            return [feature.to_dict() for feature in features]
        
        # Limited implementation for other store types
        logger.warning("list_features is limited for non-PostgreSQL offline stores")
        return []


class SecureFeatureStore:
    """Feature store with integrated security."""
    
    def __init__(self, feature_store: FeatureStore, security: Security):
        """
        Initialize the secure feature store.
        
        Args:
            feature_store: Base feature store
            security: Security module
        """
        self.feature_store = feature_store
        self.security = security
    
    def save_feature(
        self,
        feature: FeatureValue,
        user_id: int,
        username: str,
        ip_address: Optional[str] = None,
        store_type: str = "both"
    ) -> bool:
        """
        Save a feature value with security checks.
        
        Args:
            feature: Feature value to save
            user_id: ID of the user saving the feature
            username: Username of the user
            ip_address: IP address of the user
            store_type: Where to save ("online", "offline", or "both")
            
        Returns:
            True if successful, False otherwise
        """
        # Check if user has permission to create features
        has_permission = self.security.authz.check_permission(
            user_id=user_id,
            resource="feature",
            action="create"
        )
        
        if not has_permission:
            logger.warning(f"User {username} (ID: {user_id}) does not have permission to create features")
            return False
        
        try:
            # Save the feature
            self.feature_store.save_feature(feature, store_type)
            
            # Log the action
            self.security.audit.log_event(
                action="save_feature",
                resource="feature",
                resource_id=f"{feature.feature_name}/{feature.entity_id}",
                user_id=user_id,
                username=username,
                details={
                    "entity_id": feature.entity_id,
                    "feature_name": feature.feature_name,
                    "timestamp": feature.timestamp.isoformat(),
                    "store_type": store_type
                },
                ip_address=ip_address
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to save feature: {e}")
            return False
    
    def get_online_feature(
        self,
        entity_id: str,
        feature_name: str,
        user_id: int,
        username: str,
        ip_address: Optional[str] = None
    ) -> Optional[FeatureValue]:
        """
        Get a feature value from the online store with security checks.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            user_id: ID of the user retrieving the feature
            username: Username of the user
            ip_address: IP address of the user
            
        Returns:
            Feature value or None if not found or no permission
        """
        # Check if user has permission to view features
        has_permission = self.security.authz.check_permission(
            user_id=user_id,
            resource="feature",
            action="view"
        )
        
        if not has_permission:
            logger.warning(f"User {username} (ID: {user_id}) does not have permission to view features")
            return None
        
        try:
            # Get the feature
            feature = self.feature_store.get_online_feature(entity_id, feature_name)
            
            if feature:
                # Log the action
                self.security.audit.log_event(
                    action="get_feature",
                    resource="feature",
                    resource_id=f"{feature_name}/{entity_id}",
                    user_id=user_id,
                    username=username,
                    details={
                        "entity_id": entity_id,
                        "feature_name": feature_name,
                        "store_type": "online"
                    },
                    ip_address=ip_address
                )
            
            return feature
        except Exception as e:
            logger.error(f"Failed to get online feature: {e}")
            return None
    
    def get_offline_feature(
        self,
        entity_id: str,
        feature_name: str,
        user_id: int,
        username: str,
        timestamp: Optional[datetime] = None,
        ip_address: Optional[str] = None
    ) -> Optional[FeatureValue]:
        """
        Get a feature value from the offline store with security checks.
        
        Args:
            entity_id: Entity ID
            feature_name: Feature name
            timestamp: Timestamp to get feature value at
            user_id: ID of the user retrieving the feature
            username: Username of the user
            ip_address: IP address of the user
            
        Returns:
            Feature value or None if not found or no permission
        """
        # Check if user has permission to view features
        has_permission = self.security.authz.check_permission(
            user_id=user_id,
            resource="feature",
            action="view"
        )
        
        if not has_permission:
            logger.warning(f"User {username} (ID: {user_id}) does not have permission to view features")
            return None
        
        try:
            # Get the feature
            feature = self.feature_store.get_offline_feature(entity_id, feature_name, timestamp)
            
            if feature:
                # Log the action
                self.security.audit.log_event(
                    action="get_feature",
                    resource="feature",
                    resource_id=f"{feature_name}/{entity_id}",
                    user_id=user_id,
                    username=username,
                    details={
                        "entity_id": entity_id,
                        "feature_name": feature_name,
                        "timestamp": timestamp.isoformat() if timestamp else None,
                        "store_type": "offline"
                    },
                    ip_address=ip_address
                )
            
            return feature
        except Exception as e:
            logger.error(f"Failed to get offline feature: {e}")
            return None
    
    def get_training_data(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        training_time: datetime,
        user_id: int,
        username: str,
        ip_address: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get training data with security checks.
        
        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names
            training_time: Reference time for training
            user_id: ID of the user retrieving the data
            username: Username of the user
            ip_address: IP address of the user
            
        Returns:
            DataFrame with training data or empty DataFrame if no permission
        """
        # Check if user has permission to view features
        has_permission = self.security.authz.check_permission(
            user_id=user_id,
            resource="feature",
            action="view"
        )
        
        if not has_permission:
            logger.warning(f"User {username} (ID: {user_id}) does not have permission to view features")
            return pd.DataFrame()
        
        try:
            # Get the training data
            df = self.feature_store.get_training_data(
                entity_ids=entity_ids,
                feature_names=feature_names,
                training_time=training_time
            )
            
            # Log the action
            self.security.audit.log_event(
                action="get_training_data",
                resource="feature",
                resource_id=",".join(feature_names),
                user_id=user_id,
                username=username,
                details={
                    "entity_ids": entity_ids if len(entity_ids) <= 10 else f"{len(entity_ids)} entities",
                    "feature_names": feature_names,
                    "training_time": training_time.isoformat(),
                    "row_count": len(df)
                },
                ip_address=ip_address
            )
            
            return df
        except Exception as e:
            logger.error(f"Failed to get training data: {e}")
            return pd.DataFrame()
    
    def save_metadata(
        self,
        feature_name: str,
        metadata: Dict[str, Any],
        user_id: int,
        username: str,
        ip_address: Optional[str] = None,
        store_type: str = "both"
    ) -> bool:
        """
        Save metadata for a feature with security checks.
        
        Args:
            feature_name: Feature name
            metadata: Metadata to save
            user_id: ID of the user saving the metadata
            username: Username of the user
            ip_address: IP address of the user
            store_type: Where to save ("online", "offline", or "both")
            
        Returns:
            True if successful, False otherwise
        """
        # Check if user has permission to create features
        has_permission = self.security.authz.check_permission(
            user_id=user_id,
            resource="feature",
            action="create"
        )
        
        if not has_permission:
            logger.warning(f"User {username} (ID: {user_id}) does not have permission to create feature metadata")
            return False
        
        try:
            # Save the metadata
            self.feature_store.save_metadata(feature_name, metadata, store_type)
            
            # Log the action
            self.security.audit.log_event(
                action="save_metadata",
                resource="feature_metadata",
                resource_id=feature_name,
                user_id=user_id,
                username=username,
                details={
                    "feature_name": feature_name,
                    "entity_type": metadata.get("entity_type"),
                    "store_type": store_type
                },
                ip_address=ip_address
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
            return False
    
    def get_metadata(
        self,
        feature_name: str,
        user_id: int,
        username: str,
        ip_address: Optional[str] = None,
        store_type: str = "offline"
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a feature with security checks.
        
        Args:
            feature_name: Feature name
            user_id: ID of the user retrieving the metadata
            username: Username of the user
            ip_address: IP address of the user
            store_type: Where to get from ("online" or "offline")
            
        Returns:
            Feature metadata or None if not found or no permission
        """
        # Check if user has permission to view features
        has_permission = self.security.authz.check_permission(
            user_id=user_id,
            resource="feature",
            action="view"
        )
        
        if not has_permission:
            logger.warning(f"User {username} (ID: {user_id}) does not have permission to view feature metadata")
            return None
        
        try:
            # Get the metadata
            metadata = self.feature_store.get_metadata(feature_name, store_type)
            
            if metadata:
                # Log the action
                self.security.audit.log_event(
                    action="get_metadata",
                    resource="feature_metadata",
                    resource_id=feature_name,
                    user_id=user_id,
                    username=username,
                    details={
                        "feature_name": feature_name,
                        "store_type": store_type
                    },
                    ip_address=ip_address
                )
            
            return metadata
        except Exception as e:
            logger.error(f"Failed to get metadata: {e}")
            return None
    
    def list_features(
        self,
        user_id: int,
        username: str,
        entity_type: Optional[str] = None,
        ip_address: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available features with security checks.
        
        Args:
            entity_type: Filter by entity type
            user_id: ID of the user listing features
            username: Username of the user
            ip_address: IP address of the user
            
        Returns:
            List of feature metadata or empty list if no permission
        """
        # Check if user has permission to view features
        has_permission = self.security.authz.check_permission(
            user_id=user_id,
            resource="feature",
            action="view"
        )
        
        if not has_permission:
            logger.warning(f"User {username} (ID: {user_id}) does not have permission to list features")
            return []
        
        try:
            # List the features
            features = self.feature_store.list_features(entity_type)
            
            # Log the action
            self.security.audit.log_event(
                action="list_features",
                resource="feature",
                resource_id="all",
                user_id=user_id,
                username=username,
                details={
                    "entity_type": entity_type,
                    "feature_count": len(features)
                },
                ip_address=ip_address
            )
            
            return features
        except Exception as e:
            logger.error(f"Failed to list features: {e}")
            return []


# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        "online_store": {
            "type": "redis",
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "offline_store": {
            "type": "postgres",
            "host": "localhost",
            "port": 5432,
            "dbname": "featurestore",
            "user": "postgres",
            "password": "postgres"
        }
    }
    
    # Create feature store
    feature_store = FeatureStore.from_config(config)
    
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
    
    # Save metadata
    feature_store.save_metadata("user_avg_purchase_30d", metadata)
    
    # Create a feature value
    feature = FeatureValue(
        entity_id="user123",
        feature_name="user_avg_purchase_30d",
        value=325.50,
        timestamp=datetime.utcnow(),
        metadata={"source": "transaction_db"}
    )
    
    # Save feature
    feature_store.save_feature(feature)
    
    # Get feature
    retrieved_feature = feature_store.get_online_feature("user123", "user_avg_purchase_30d")
    
    if retrieved_feature:
        print(f"Retrieved feature value: {retrieved_feature.value}")
    else:
        print("Feature not found")