import json
import random
import time
import datetime
from abc import ABC, abstractmethod

# ---------------------------
# Sensor Classes (Data Sources)
# ---------------------------
class Sensor(ABC):
    def __init__(self):
        self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.sensor_id = None
        self.sensor_type = None
        # Assign a fixed ZIP code to represent the sensor's location in the city.
        # Using a three-digit string, e.g., "001" to "100".
        self.zip = f"{random.randint(1, 100):03d}"

    @abstractmethod
    def generate_data(self):
        """Return sensor-specific measurements."""
        pass

    def to_dict(self):
        data = self.generate_data()
        data.update({
            "sensor_id": self.sensor_id,
            "type": self.sensor_type,
            "timestamp": self.timestamp,
            "zip": self.zip
        })
        return data


class TrafficSensor(Sensor):
    def __init__(self):
        super().__init__()
        self.sensor_type = "traffic"
        self.sensor_id = f"traffic_{random.randint(1, 100)}"
        # Create a distribution for different vehicle types.
        vehicle_types = ["sedan", "SUV", "truck", "bus", "motorcycle"]
        self.vehicle_distribution = {vt: random.randint(0, 50) for vt in vehicle_types}
        self.vehicle_count = sum(self.vehicle_distribution.values())
        self.avg_speed = round(random.uniform(20, 80), 2)

    def generate_data(self):
        return {
            "vehicle_count": self.vehicle_count,
            "avg_speed": self.avg_speed,
            "vehicle_distribution": self.vehicle_distribution
        }


class EnvironmentSensor(Sensor):
    def __init__(self):
        super().__init__()
        self.sensor_type = "environment"
        self.sensor_id = f"env_{random.randint(1, 100)}"
        self.temperature = round(random.uniform(-10, 40), 2)
        self.humidity = random.randint(20, 100)
        self.air_quality_index = random.randint(0, 500)

    def generate_data(self):
        return {
            "temperature": self.temperature,
            "humidity": self.humidity,
            "air_quality_index": self.air_quality_index
        }


class UtilitySensor(Sensor):
    def __init__(self):
        super().__init__()
        self.sensor_type = "utility"
        self.sensor_id = f"utility_{random.randint(1, 100)}"
        self.consumption = round(random.uniform(0, 2000), 2)
        self.voltage = round(random.uniform(110, 240), 2)
        self.current = round(random.uniform(0, 50), 2)

    def generate_data(self):
        return {
            "consumption": self.consumption,
            "voltage": self.voltage,
            "current": self.current
        }


class WaterSensor(Sensor):
    def __init__(self):
        super().__init__()
        self.sensor_type = "water"
        self.sensor_id = f"water_{random.randint(1, 100)}"
        self.pH = round(random.uniform(6.5, 8.5), 2)
        self.turbidity = round(random.uniform(0.1, 5.0), 2)
        self.contaminant_level = random.randint(0, 100)

    def generate_data(self):
        return {
            "pH": self.pH,
            "turbidity": self.turbidity,
            "contaminant_level": self.contaminant_level
        }

# List of sensor types to simulate.
SENSOR_CLASSES = [TrafficSensor, EnvironmentSensor, UtilitySensor, WaterSensor]


# ---------------------------
# Helper: Introduce Rare Unusual Data
# ---------------------------
def maybe_introduce_unusual_data(sensor_data, unusual_probability=0.05, min_multiplier=2, max_multiplier=5):
    """
    With low probability, modify one measurement to be unusual,
    simulating a rare unexpected event in the data stream.
    """
    if random.random() < unusual_probability:
        sensor_type = sensor_data.get("type")
        field_map = {
            "traffic": ["vehicle_count", "avg_speed"],
            "environment": ["temperature", "humidity", "air_quality_index"],
            "utility": ["consumption", "voltage", "current"],
            "water": ["pH", "turbidity", "contaminant_level"]
        }
        if sensor_type in field_map:
            key = random.choice(field_map[sensor_type])
            original_value = sensor_data.get(key)
            if isinstance(original_value, (int, float)):
                multiplier = random.uniform(min_multiplier, max_multiplier)
                sensor_data[key] = type(original_value)(round(original_value * multiplier, 2))
                # Tag the event as unusual.
                sensor_data["unusual"] = True
    return sensor_data


def simulate_sensor_reading(unusual_probability=0.05, min_multiplier=2, max_multiplier=5):
    """
    Creates a sensor reading by instantiating one random sensor type
    and then optionally introducing an unusual value.
    """
    sensor = random.choice(SENSOR_CLASSES)()
    data = sensor.to_dict()
    data = maybe_introduce_unusual_data(data, unusual_probability, min_multiplier, max_multiplier)
    return data


# ---------------------------
# Simulated ML Pipeline Components (Drift Detection & Model Training)
# ---------------------------
def train_and_serve_model():
    """
    Simulate retraining the ML model and redeploying it,
    triggered when data drift is detected.
    """
    print("DRIFT ALERT: Unusual data drift detected. Initiating model retraining and redeployment.")


# ---------------------------
# Real-Time Streaming Simulation
# ---------------------------
def run_streaming_pipeline(num_events=100, unusual_probability=0.05, min_multiplier=2,
                           max_multiplier=5, stream_interval=1):
    """
    Simulate a real-time streaming ML platform with the following features:
      - Real-time ingestion of sensor data from multiple sources.
      - Stream processing with drift detection.
      - ML model retraining and redeployment upon drift.
      - Monitoring and alerting of unusual events.
    
    Each sensor reading now includes a fixed ZIP code in the city for location-specific analysis.
    """
    state = {
        "total_events": 0,
        "window": []  # Sliding window for drift detection.
    }
    drift_threshold = 0.10  # Trigger drift if more than 10% of events in window are unusual.
    window_size = 50        # Sliding window size.

    for i in range(num_events):
        event = simulate_sensor_reading(unusual_probability, min_multiplier, max_multiplier)
        state["total_events"] += 1
        state["window"].append(event)

        event_status = "UNUSUAL" if event.get("unusual") else "Normal"
        print(f"[Event {state['total_events']}] Type: {event_status} | Data: {json.dumps(event)}")

        # Drift detection: Process the sliding window once its size reaches the designated value.
        if len(state["window"]) >= window_size:
            unusual_count = sum(1 for e in state["window"] if e.get("unusual"))
            unusual_ratio = unusual_count / window_size
            if unusual_ratio > drift_threshold:
                print(f"Drift Detection: {unusual_count}/{window_size} events unusual ({unusual_ratio*100:.2f}%).")
                train_and_serve_model()
                state["window"] = []  # Reset the window after drift handling.
        time.sleep(stream_interval)


# ---------------------------
# Main Entry Point
# ---------------------------
if __name__ == "__main__":
    print("Starting Real-Time Data Streaming ML Platform Simulation...\n")
    run_streaming_pipeline(num_events=100, unusual_probability=0.05, stream_interval=1)
