#!/usr/bin/env python3
"""
Command-line interface for the LLM-powered data simulation system.
This script allows users to generate synthetic data with configurable drift patterns.
"""

import os
import sys
import argparse
import logging
import yaml
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_powered.base_generator import BaseDataGenerator
from llm_powered.llm_generator import LLMDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('drift_simulation.log')
    ]
)

logger = logging.getLogger('drift_simulation')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Generate synthetic data with drift patterns for ML testing')
    
    parser.add_argument('-c', '--config', type=str, default='./src/simulation/llm_powered/config.yaml',
                        help='Path to the configuration file')
    
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for generated data (overrides config)')
    
    parser.add_argument('-n', '--num-samples', type=int, default=None,
                        help='Total number of samples to generate (overrides config)')
    
    parser.add_argument('-b', '--batch-size', type=int, default=None,
                        help='Batch size for generation (overrides config)')
    
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    parser.add_argument('--use-llm', action='store_true', default=True,
                        help='Use LLM for enhanced data generation')
    
    parser.add_argument('--no-llm', dest='use_llm', action='store_false',
                        help='Disable LLM for data generation (use base generator only)')
    
    parser.add_argument('--apply-drift', type=str, default=None, choices=['random', 'all'],
                        help='Apply drift scenarios: random=one random scenario, all=all scenarios sequentially')
    
    parser.add_argument('-d', '--drift-scenario', type=str, default=None,
                        help='Name of specific drift scenario to apply')
    
    parser.add_argument('--kafka', action='store_true', default=False,
                        help='Output data to Kafka topic (for streaming tests)')
    
    parser.add_argument('--kafka-topic', type=str, default='drift-ml-data',
                        help='Kafka topic for streaming data')
    
    parser.add_argument('--kafka-bootstrap-servers', type=str, default='localhost:9092',
                        help='Kafka bootstrap servers')
    
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose logging')
    
    return parser.parse_args()

def update_config_with_args(config, args):
    """Update configuration with command line arguments."""
    # Update simulation settings
    if args.seed is not None:
        config['simulation']['seed'] = args.seed
    
    if args.output is not None:
        config['simulation']['output_path'] = args.output
    
    # Update data generation settings
    if args.num_samples is not None:
        config['data_generation']['total_samples'] = args.num_samples
    
    if args.batch_size is not None:
        config['data_generation']['batch_size'] = args.batch_size
    
    return config

def select_drift_scenario(config, args):
    """Select drift scenario to apply based on arguments."""
    drift_scenarios = config.get('drift_scenarios', [])
    
    if not drift_scenarios:
        logger.warning("No drift scenarios defined in configuration")
        return None
    
    if args.drift_scenario:
        # Find scenario by name
        for scenario in drift_scenarios:
            if scenario['name'] == args.drift_scenario:
                logger.info(f"Selected drift scenario: {scenario['name']}")
                return scenario
        
        logger.warning(f"Drift scenario '{args.drift_scenario}' not found")
        return None
    
    elif args.apply_drift == 'random':
        # Select random scenario
        import random
        scenario = random.choice(drift_scenarios)
        logger.info(f"Selected random drift scenario: {scenario['name']}")
        return scenario
    
    elif args.apply_drift == 'all':
        # Return all scenarios (caller will handle sequentially)
        logger.info(f"Will apply all {len(drift_scenarios)} drift scenarios")
        return drift_scenarios
    
    return None

def send_to_kafka(df, args):
    """Send data to Kafka topic."""
    try:
        from kafka import KafkaProducer
        import json
        
        logger.info(f"Connecting to Kafka at {args.kafka_bootstrap_servers}")
        producer = KafkaProducer(
            bootstrap_servers=args.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        for _, row in df.iterrows():
            # Convert row to dictionary and send to Kafka
            record = row.to_dict()
            producer.send(args.kafka_topic, record)
        
        producer.flush()
        logger.info(f"Sent {len(df)} records to Kafka topic '{args.kafka_topic}'")
    
    except ImportError:
        logger.error("Kafka Python client not installed. Run 'pip install kafka-python'")
    except Exception as e:
        logger.error(f"Error sending data to Kafka: {str(e)}")

def main():
    """Main entry point for the CLI."""
    args = parse_arguments()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        return 1
    
    # Update config with command line arguments
    config = update_config_with_args(config, args)
    
    # Ensure output directory exists
    output_path = config['simulation']['output_path']
    os.makedirs(output_path, exist_ok=True)
    
    # Select appropriate generator
    if args.use_llm:
        logger.info("Using LLM-powered data generator")
        generator = LLMDataGenerator(args.config)
    else:
        logger.info("Using base data generator")
        generator = BaseDataGenerator(args.config)
    
    # Select drift scenario
    drift_scenario = select_drift_scenario(config, args)
    
    # Generate data
    if isinstance(drift_scenario, list):
        # Apply all scenarios sequentially
        all_data = []
        for i, scenario in enumerate(drift_scenario):
            logger.info(f"Generating data with drift scenario {i+1}/{len(drift_scenario)}: {scenario['name']}")
            
            # Update generator's current drift scenario
            generator.current_drift_scenario = scenario
            
            # Generate a portion of the data with this scenario
            total_samples = config['data_generation']['total_samples']
            scenario_samples = total_samples // len(drift_scenario)
            
            # Override total samples and generate
            config['data_generation']['total_samples'] = scenario_samples
            batch_data = generator.generate_all_data()
            
            # Add scenario metadata
            batch_data['drift_scenario'] = scenario['name']
            batch_data['drift_type'] = scenario['type']
            
            all_data.append(batch_data)
        
        # Combine all data
        if all_data:
            df = pd.concat(all_data)
            
            # Save combined data
            output_file = os.path.join(output_path, f"synthetic_data_all_scenarios.csv")
            df.to_csv(output_file, index=False)
            logger.info(f"Saved combined data to {output_file}")
            
            # Send to Kafka if requested
            if args.kafka:
                send_to_kafka(df, args)
    else:
        # Generate with single scenario or no drift
        if drift_scenario:
            generator.current_drift_scenario = drift_scenario
            logger.info(f"Generating data with drift scenario: {drift_scenario['name']}")
        else:
            logger.info("Generating data with no specific drift scenario")
        
        df = generator.generate_all_data()
        
        # Send to Kafka if requested
        if args.kafka:
            send_to_kafka(df, args)
    
    logger.info("Data generation complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())