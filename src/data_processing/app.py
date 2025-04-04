"""
Main entry point for the Flink data processing application.
"""
import os
import sys
import json
import logging
import argparse
from typing import Dict, Any, List
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment, EnvironmentSettings

from src.data_processing.flink_processor import FlinkProcessor
from src.data_processing.transformations import (
    normalize_features,
    filter_invalid_data,
    enrich_with_metadata
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Drift-ML Data Processor")
    parser.add_argument(
        "--bootstrap-servers",
        type=str,
        default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
        help="Kafka bootstrap servers"
    )
    parser.add_argument(
        "--input-topics",
        type=str,
        default=os.environ.get("INPUT_TOPICS", "input-data"),
        help="Comma-separated list of input topics"
    )
    parser.add_argument(
        "--output-topic",
        type=str,
        default=os.environ.get("OUTPUT_TOPIC", "processed-data"),
        help="Output topic"
    )
    parser.add_argument(
        "--error-topic",
        type=str,
        default=os.environ.get("ERROR_TOPIC", "error-data"),
        help="Error topic"
    )
    parser.add_argument(
        "--group-id",
        type=str,
        default=os.environ.get("GROUP_ID", "drift-ml-processor"),
        help="Consumer group ID"
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default=os.environ.get("JOB_NAME", "drift-ml-data-processor"),
        help="Flink job name"
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=int(os.environ.get("FLINK_PARALLELISM", "4")),
        help="Flink parallelism"
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse arguments
    args = parse_args()
    
    # Set up Flink environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(args.parallelism)
    
    # Configure checkpointing for fault tolerance
    env.enable_checkpointing(60000)  # 60 seconds
    env.get_checkpoint_config().set_min_pause_between_checkpoints(30000)  # 30 seconds
    env.get_checkpoint_config().set_checkpoint_timeout(20000)  # 20 seconds
    
    # Create processor
    processor = FlinkProcessor(
        bootstrap_servers=args.bootstrap_servers,
        input_topics=args.input_topics.split(','),
        output_topic=args.output_topic,
        error_topic=args.error_topic,
        group_id=args.group_id,
        job_name=args.job_name
    )
    
    # Configure processor with custom transformations
    processor.add_filter(filter_invalid_data)
    processor.add_transformation(normalize_features)
    processor.add_transformation(enrich_with_metadata)
    
    # Build and execute pipeline
    processor.build_pipeline()
    processor.execute()

if __name__ == "__main__":
    main()
