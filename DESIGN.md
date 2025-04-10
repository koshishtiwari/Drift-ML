# Real-Time Data Streaming ML Platform

<!--DESIGN.md -->

## 1. System Architecture Overview

```
+------------------+     +---------------------+     +-------------------+
| DATA SOURCES     |     | DATA PROCESSING     |     | STORAGE           |
|                  |     |                     |     |                   |
| - IoT Sensors    |---->| - Stream Processing |---->| - Time Series DB  |
| - Market Data    |     | - ETL Pipelines     |     | - Data Lake       |
| - News Scraping  |     | - Schema Registry   |     | - Feature Store   |
+------------------+     +---------------------+     +-------------------+
          |                        |                         |
          v                        v                         v
+------------------+     +---------------------+     +-------------------+
| ML PLATFORM      |     | MONITORING          |     | RESULTS & OUTPUT  |
|                  |     |                     |     |                   |
| - Model Training |<--->| - Data Monitoring   |<--->| - Dashboards      |
| - Serving        |     | - Model Monitoring  |     | - Alerting        |
| - Drift Detection|     | - System Metrics    |     | - API Endpoints   |
+------------------+     +---------------------+     +-------------------+
```

## 2. Data Sources & Ingestion

### 2.1 IoT Sensor Data
- **Sources**: Traffic sensors, environmental monitors, utility meters, etc.
- **Protocol**: MQTT for lightweight messaging
- **Ingestion Rate**: High-frequency, small payloads (ms to seconds)

```
For now,
**Simulator Component**:
Use LLM to create simulated data

%pip install -U -q "google-genai>=1.0.0"


import os
import random
import json
import datetime
from google import genai

def simulate_sensor_data():
    """
    Generates simulated IoT sensor data using random values.
    The sensor can be one of the following types:
      - Traffic sensor: vehicle_count, avg_speed, and location (lat, lon)
      - Environmental monitor: temperature, humidity, air_quality_index
      - Utility meter: consumption, voltage, current
    """
    sensor_category = random.choice(["traffic", "environment", "utility"])
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()    
    if sensor_category == "traffic":
        data = {
            "type": "traffic",
            "sensor_id": f"traffic_{random.randint(1, 5)}",
            "timestamp": timestamp,
            "vehicle_count": random.randint(0, 100),
            "avg_speed": round(random.uniform(20, 80), 2),
            "location": {
                "lat": round(random.uniform(-90, 90), 4),
                "lon": round(random.uniform(-180, 180), 4)
            }
        }
    elif sensor_category == "environment":
        data = {
            "type": "environment",
            "sensor_id": f"env_{random.randint(1, 5)}",
            "timestamp": timestamp,
            "temperature": round(random.uniform(-10, 40), 2),
            "humidity": random.randint(20, 100),
            "air_quality_index": random.randint(0, 500)
        }
    else:  # utility sensor
        data = {
            "type": "utility",
            "sensor_id": f"utility_{random.randint(1, 5)}",
            "timestamp": timestamp,
            "consumption": round(random.uniform(0, 2000), 2),
            "voltage": round(random.uniform(110, 240), 2),
            "current": round(random.uniform(0, 50), 2)
        }
    return data

def generate_sensor_data_with_llm():
    """
    Uses Google GenAI to generate a simulated IoT sensor JSON payload.
    If the output cannot be parsed as valid JSON, it falls back to using random generation.
    """
    client = genai.Client()  # Ensure your API key is set via environment variables

    prompt = (
        "Generate a valid JSON object representing a simulated IoT sensor reading from one of these categories: "
        "traffic sensor, environmental monitor, or utility meter. For a traffic sensor, include sensor_id (string), "
        "timestamp (ISO 8601 format), vehicle_count (integer), avg_speed (float), and location (object with lat and lon as floats). "
        "For an environmental monitor, include sensor_id, timestamp, temperature (float), humidity (integer), and air_quality_index (integer). "
        "For a utility meter, include sensor_id, timestamp, consumption (float), voltage (float), and current (float). "
        "Randomly choose one category and ensure the JSON is properly formatted."
    )

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )

    try:
        sensor_json = json.loads(response.text)
    except Exception as e:
        print("LLM output parsing failed, falling back to random generation. Error:", e)
        sensor_json = simulate_sensor_data()
    
    return sensor_json

if __name__ == "__main__":
    # Example: Generating simulated IoT sensor data using the random generator.
    print("Simulated sensor data (random generation):")
    random_data = simulate_sensor_data()
    print(json.dumps(random_data, indent=4))
    
    print("\nSimulated sensor data (using LLM):")
    llm_data = generate_sensor_data_with_llm()
    print(json.dumps(llm_data, indent=4))

```

### 2.2 Market Data
- **Sources**: Stock markets, commodities, real estate indices
- **Protocol**: WebSockets for real-time data, REST API for historical
- **Ingestion Rate**: High frequency (Real-time to Historical (1y/3y/5y/10y))
```
Use Alpaca API for real-time and historical US stocks
Make it modular so that, we could add redundant/ fall back sources
```

### 2.3 Web Scraping
- **Sources**: Location based news websites, social media, government announcements
- **Protocol**: HTTP/HTTPS
- **Ingestion Rate**: Low frequency (minutes to hours)
- **Simulator Component**:

```python
Use agentic ai LLM to do web search

Use this,

%pip install -U -q google-genai requests

import os
import requests
from google import genai

def fetch_news(topic, api_key, page_size=5):
    """
    Fetches the latest news articles for a given topic using NewsAPI.
    Returns a concatenated string with title, description, and URL for each article.
    """
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": topic,
        "pageSize": page_size,
        "apiKey": api_key,
        "language": "en"
    }
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        raise Exception(f"News API returned error {response.status_code}: {response.text}")

    data = response.json()
    articles = data.get('articles', [])
    
    if not articles:
        return "No articles found."
    
    collected_data = ""
    for article in articles:
        title = article.get("title", "No Title")
        description = article.get("description", "No Description")
        url = article.get("url", "No URL")
        article_summary = f"Title: {title}\nDescription: {description}\nURL: {url}\n\n"
        collected_data += article_summary
    return collected_data

def summarize_news(news_content, model="gemini-2.0-flash"):
    """
    Uses Google GenAI to summarize the fetched news content.
    """
    # Initialize the GenAI client (make sure your API key is properly configured in your environment)
    client = genai.Client()

    # Create a prompt asking the model to summarize the news content
    prompt = (
        "Summarize the following news articles for me in a concise, clear paragraph, "
        "highlighting the main events and general sentiment:\n\n"
        f"{news_content}\n"
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )

    return response.text

def main():
    # Obtain the NewsAPI key from environment variables.
    news_api_key = os.getenv("NEWS_API_KEY")
    if not news_api_key:
        print("Error: Please set the NEWS_API_KEY environment variable.")
        return

    # Ask the user for the news topic they are interested in.
    topic = input("Enter a news topic: ")

    print("\nFetching news articles...")
    try:
        news_content = fetch_news(topic, news_api_key)
        print("News articles fetched:\n")
        print(news_content)
    except Exception as e:
        print(f"An error occurred while fetching news: {e}")
        return

    print("\nSummarizing news with Google GenAI...\n")
    try:
        summary = summarize_news(news_content)
        print("Summary of Latest News:")
        print(summary)
    except Exception as e:
        print(f"An error occurred while summarizing news: {e}")

if __name__ == "__main__":
    main()
```

## 3. Data Processing & Flow

### 3.1 Stream Processing Framework
- **Technology**: Apache Kafka + Kafka Streams / Apache Flink
- **Functions**: 
  - Real-time data transformation
  - Windowing operations
  - Enrichment with contextual data
  - Anomaly detection

```java
// Example Kafka Streams Application
import org.apache.kafka.common.serialization.Serdes;
import org.apache.kafka.streams.KafkaStreams;
import org.apache.kafka.streams.StreamsBuilder;
import org.apache.kafka.streams.StreamsConfig;
import org.apache.kafka.streams.kstream.*;

import java.util.Properties;
import java.time.Duration;

public class IoTStreamProcessor {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "iot-stream-processor");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass().getName());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass().getName());

        StreamsBuilder builder = new StreamsBuilder();
        
        // Read from IoT sensors topic
        KStream<String, String> sensorStream = builder.stream("city.sensors");
        
        // Process traffic sensors data
        KStream<String, String> trafficStream = sensorStream
            .filter((key, value) -> value.contains("traffic"))
            .mapValues(value -> {
                // Parse JSON, extract and transform data
                // This is simplified pseudo-code
                return transformTrafficData(value);
            });
        
        // Calculate average traffic by location in 5-minute windows
        KTable<Windowed<String>, String> trafficByLocation = trafficStream
            .groupByKey()
            .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
            .aggregate(
                () -> "0,0", // initializer (count, sum)
                (key, value, aggregate) -> updateAggregate(aggregate, value),
                Materialized.with(Serdes.String(), Serdes.String())
            );
        
        // Output processed data to new topic
        trafficByLocation
            .toStream()
            .to("city.processed.traffic");
        
        KafkaStreams streams = new KafkaStreams(builder.build(), props);
        streams.start();
        
        // Add shutdown hook
        Runtime.getRuntime().addShutdownHook(new Thread(streams::close));
    }
    
    private static String transformTrafficData(String rawData) {
        // Implementation details
        return transformedData;
    }
    
    private static String updateAggregate(String aggregate, String newValue) {
        // Implementation details
        return updatedAggregate;
    }
}
```

### 3.2 Schema Registry
- **Technology**: Confluent Schema Registry / AWS Glue Schema Registry
- **Purpose**: 
  - Schema evolution management
  - Data compatibility validation
  - Centralized schema storage

### 3.3 Data Quality & Validation
- **Techniques**:
  - Schema validation
  - Statistical validation (range checks, distributions)
  - Cross-field validation
  - Completeness checks

```python
# Example Data Validation 
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, isnull
import great_expectations as ge

def validate_iot_data(batch_df):
    # Initialize Spark session
    spark = SparkSession.builder.appName("IoT Data Validation").getOrCreate()
    
    # Load batch data
    df = spark.createDataFrame(batch_df)
    
    # Basic validation checks
    validation_results = df.select(
        count("*").alias("total_records"),
        count(when(isnull("sensor_id") | (col("sensor_id") == ""), 1)).alias("missing_sensor_id"),
        count(when(isnull("timestamp") | (col("timestamp") == ""), 1)).alias("missing_timestamp"),
        count(when(isnull("value") | isnan("value"), 1)).alias("missing_value")
    ).collect()[0]
    
    # Calculate error rates
    error_rates = {
        "missing_sensor_id_rate": validation_results.missing_sensor_id / validation_results.total_records,
        "missing_timestamp_rate": validation_results.missing_timestamp / validation_results.total_records,
        "missing_value_rate": validation_results.missing_value / validation_results.total_records
    }
    
    # Advanced validation with Great Expectations
    ge_df = ge.dataset.SparkDFDataset(df)
    
    # Add expectations based on data type
    expectations_results = {}
    
    # Check value ranges by sensor type
    expectations_results["traffic_range"] = ge_df.expect_column_values_to_be_between(
        "value", 
        min_value=0, 
        max_value=500,
        condition_parser="pandas",
        row_condition="type == 'traffic'"
    )
    
    expectations_results["temperature_range"] = ge_df.expect_column_values_to_be_between(
        "value", 
        min_value=-20, 
        max_value=50,
        condition_parser="pandas",
        row_condition="type == 'temperature'"
    )
    
    # Return validation summary
    return {
        "basic_validation": error_rates,
        "advanced_validation": expectations_results,
        "overall_quality_score": calculate_quality_score(error_rates, expectations_results)
    }

def calculate_quality_score(error_rates, expectations_results):
    # Implementation details
    return quality_score
```

## 4. Storage Architecture

### 4.1 Time Series Database
- **Technology**: questDB
- **Purpose**: Store high-volume data with efficient time-based queries
- **Schema Design**:
  - Measurements: sensor readings
  - Tags: sensor_id, type,  location, time
  - Fields: value, metadata
  - Retention policies

### 4.2 Data Lake
- **Technology**: Delta Lake
- **Organization**:
  - Bronze layer: raw data
  - Silver layer: validated, cleansed data
  - Gold layer: aggregated, enriched data
- **Access Patterns**: 
  - Batch processing
  - Historical analysis
  - Training data preparation

### 4.3 Feature Store
- **Technology**: Feast
- **Capabilities**:
  - Online/offline feature serving
  - Feature versioning
  - Point-in-time correctness
  - Feature sharing across models

```python
# Feature Store Definition Example (using Feast)
from datetime import datetime, timedelta
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource

# Define entity
# Define data sources
# Define feature views
```

## 5. ML Platform

### 5.1 Model Training Pipeline
- **Technology**: MLflow / Kubeflow / SageMaker
- **Capabilities**:
  - Experiment tracking
  - Hyperparameter optimization
  - Model registry
  - Pipeline orchestration

```python
# ML Model Training Pipeline (with MLflow)
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def train_traffic_prediction_model(data_path, feature_list, target="congestion_level"):
    # Start MLflow experiment
    mlflow.set_experiment("traffic_prediction")
    
    # Load and prepare data
    data = pd.read_parquet(data_path)
    X = data[feature_list]
    y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define model parameters
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42
    }
    
    # Train model with MLflow tracking
    with mlflow.start_run():
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** 0.5
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        
        # Log feature importance
        feature_importance = pd.DataFrame(
            model.feature_importances_,
            index=X.columns,
            columns=["importance"]
        ).sort_values("importance", ascending=False)
        
        # Log model
        mlflow.sklearn.log_model(model, "model", registered_model_name="traffic_prediction")
        
        return model, feature_importance, rmse
```

### 5.2 Serving Infrastructure
- **Technology**: TensorFlow Serving / Seldon Core / KServe
- **Deployment Patterns**:
  - Canary deployments
  - A/B testing
  - Shadow mode
- **Serving Types**:
  - Online prediction (low latency)
  - Batch prediction (high throughput)

### 5.3 Model Monitoring & Drift Detection
- **Types of Drift**:
  - Feature drift
  - Concept drift
  - Data quality drift
- **Monitoring Metrics**:
  - Statistical distance measures (KL divergence, JS distance)
  - Performance metrics (accuracy, precision, recall)
  - Prediction distribution changes

```python
# Drift Detection Example
import numpy as np
from scipy.stats import ks_2samp
import pandas as pd
from datetime import datetime, timedelta

class DriftDetector:
    def __init__(self, reference_data, drift_threshold=0.05):
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.feature_distributions = self._calculate_distributions(reference_data)
        
    def _calculate_distributions(self, data):
        distributions = {}
        for column in data.columns:
            if data[column].dtype in [np.float64, np.int64]:
                distributions[column] = {
                    "mean": data[column].mean(),
                    "std": data[column].std(),
                    "min": data[column].min(),
                    "max": data[column].max(),
                    "median": data[column].median(),
                    "histogram": np.histogram(data[column], bins=10)
                }
        return distributions
    
    def detect_drift(self, current_data):
        drift_results = {}
        overall_drift = False
        
        for column, ref_dist in self.feature_distributions.items():
            if column in current_data.columns and current_data[column].dtype in [np.float64, np.int64]:
                # Perform Kolmogorov-Smirnov test
                ks_result = ks_2samp(
                    self.reference_data[column].dropna(), 
                    current_data[column].dropna()
                )
                
                drift_results[column] = {
                    "p_value": ks_result.pvalue,
                    "statistic": ks_result.statistic,
                    "drift_detected": ks_result.pvalue < self.drift_threshold,
                    "mean_difference": abs(ref_dist["mean"] - current_data[column].mean()) / ref_dist["std"] if ref_dist["std"] > 0 else 0
                }
                
                if drift_results[column]["drift_detected"]:
                    overall_drift = True
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_drift_detected": overall_drift,
            "feature_drift": drift_results,
            "sample_size": len(current_data)
        }
    
    def update_reference(self, new_reference_data):
        self.reference_data = new_reference_data
        self.feature_distributions = self._calculate_distributions(new_reference_data)
```

## 6. Monitoring & Observability

### 6.1 System Monitoring
- **Technology**: Prometheus / Grafana / ELK Stack
- **Metrics**:
  - Resource utilization (CPU, memory, disk, network)
  - Throughput (messages/second)
  - Latency (processing time)
  - Error rates

### 6.2 Data Monitoring
- **Metrics**:
  - Volume (records/second)
  - Schema violations
  - Data quality scores
  - Freshness/timeliness

### 6.3 Model Monitoring
- **Metrics**:
  - Prediction latency
  - Feature distribution
  - Model performance
  - Drift indicators

```python
# Monitoring Dashboard Setup (Prometheus + Grafana)
from prometheus_client import start_http_server, Summary, Counter, Gauge
import random
import time

# Create metrics
PREDICTION_LATENCY = Summary('prediction_latency_seconds', 'Time taken for prediction')
REQUESTS = Counter('prediction_requests_total', 'Total prediction requests')
ERROR_RATE = Counter('prediction_errors_total', 'Total prediction errors')
DATA_DRIFT = Gauge('data_drift_score', 'Current data drift score')
MODEL_PERFORMANCE = Gauge('model_performance_score', 'Current model performance metric')

# Example instrumentation wrapper
def monitor_prediction_service(predict_fn):
    def wrapper(input_data):
        REQUESTS.inc()
        try:
            with PREDICTION_LATENCY.time():
                result = predict_fn(input_data)
            return result
        except Exception as e:
            ERROR_RATE.inc()
            raise e
    return wrapper

# Update drift and performance metrics
def update_monitoring_metrics(drift_score, performance_score):
    DATA_DRIFT.set(drift_score)
    MODEL_PERFORMANCE.set(performance_score)

# Start prometheus metrics server
def start_metrics_server(port=8000):
    start_http_server(port)
    print(f"Prometheus metrics server started on port {port}")
```

## 7. Results Storage & Reporting

### 7.1 Prediction Storage
- **Technology**: MongoDB / PostgreSQL / Elasticsearch
- **Schema**: 
  - Prediction ID
  - Input features
  - Output prediction
  - Confidence/probability
  - Timestamp
  - Model version

### 7.2 Visualization & Dashboards
- **Technology**: Grafana / Superset / Tableau
- **Dashboard Types**:
  - Operational dashboards
  - Analytics dashboards
  - Executive dashboards

### 7.3 Alerting System
- **Technology**: Alertmanager / PagerDuty / Opsgenie
- **Alert Types**:
  - System alerts (component failures, resource exhaustion)
  - Data alerts (quality issues, volume anomalies)
  - Model alerts (performance degradation, drift detected)

```python
# Alerting System Example
import json
import requests
from datetime import datetime

class AlertManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.alert_endpoints = self.config["endpoints"]
        self.alert_rules = self.config["rules"]
        self.alert_history = []
    
    def check_threshold(self, metric_name, value):
        if metric_name in self.alert_rules:
            rule = self.alert_rules[metric_name]
            
            if rule["comparison"] == "greater_than" and value > rule["threshold"]:
                return True
            elif rule["comparison"] == "less_than" and value < rule["threshold"]:
                return True
            elif rule["comparison"] == "equals" and value == rule["threshold"]:
                return True
        
        return False
    
    def trigger_alert(self, metric_name, value, additional_info=None):
        if not self.check_threshold(metric_name, value):
            return False
        
        rule = self.alert_rules[metric_name]
        severity = rule["severity"]
        
        alert = {
            "timestamp": datetime.now().isoformat(),
            "metric": metric_name,
            "value": value,
            "threshold": rule["threshold"],
            "comparison": rule["comparison"],
            "severity": severity,
            "message": rule["message_template"].format(
                metric=metric_name,
                value=value,
                threshold=rule["threshold"]
            ),
            "additional_info": additional_info
        }
        
        # Store alert in history
        self.alert_history.append(alert)
        
        # Send alert to appropriate endpoints based on severity
        for endpoint in self.alert_endpoints:
            if severity in endpoint["severities"]:
                self._send_alert(endpoint, alert)
        
        return True
    
    def _send_alert(self, endpoint, alert):
        if endpoint["type"] == "webhook":
            requests.post(
                endpoint["url"],
                json=alert,
                headers={"Content-Type": "application/json"}
            )
        elif endpoint["type"] == "email":
            # Email sending implementation
            pass
        elif endpoint["type"] == "sms":
            # SMS sending implementation
            pass
```

## 8. Integration Architecture

### 8.1 API Gateway
- **Technology**: Kong / AWS API Gateway / Nginx
- **Functions**:
  - Authentication/Authorization
  - Rate limiting
  - Request routing
  - Response transformation

### 8.2 Deployment & DevOps
- **Technology**: Kubernetes / Terraform / GitHub Actions
- **Practices**:
  - Infrastructure as Code
  - CI/CD pipelines
  - Blue/Green deployments
  - Automated testing

### 8.3 Security & Compliance
- **Considerations**:
  - Data encryption (in-transit, at-rest)
  - Access control
  - Audit logging
  - Privacy compliance (GDPR, CCPA)

## 9. Implementation Roadmap

### Phase 1: Core Infrastructure
- Set up data collection simulators
- Deploy Kafka and Schema Registry
- Implement basic stream processing
- Deploy time-series database

### Phase 2: ML Foundation
- Implement feature store
- Set up MLflow for experiment tracking
- Create basic prediction models
- Develop model serving infrastructure

### Phase 3: Monitoring & Observability
- Deploy Prometheus/Grafana stack
- Implement data quality monitoring
- Add drift detection capabilities
- Set up alerting system

### Phase 4: Advanced Features
- Implement automated retraining
- Add A/B testing capabilities
- Enhance visualization dashboards
- Improve alerting with anomaly detection

## 10. Technology Stack Summary

| Component | Recommended Technologies |
|-----------|--------------------------|
| Data Ingestion | MQTT, Kafka, Websockets |
| Stream Processing | Kafka Streams, Flink |
| Storage | questDB, Delta Lake, Redis |
| ML Platform | MLflow, Kubeflow, Seldon |
| Monitoring | Prometheus, Grafana, ELK |
| Orchestration | Airflow, Kubernetes |
| Visualization | Grafana, Superset |
| Deployment | Terraform, GitHub Actions |