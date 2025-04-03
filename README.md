# Drift-ML Platform

Drift-ML is a streamlined real-time machine learning platform with integrated security features.

## Components
- **Data Ingestion**: Apache Kafka for streaming data
- **Data Processing**: Apache Flink for stream processing
- **Feature Store**: Redis (online) and PostgreSQL (offline)
- **Model Training**: MLflow-based training
- **Model Registry**: MLflow for model versioning and tracking
- **Model Serving**: FastAPI-based model server with security
- **Monitoring**: Prometheus and Grafana
- **Security**: Authentication, Authorization, Encryption, and Audit Logging

## Running with Docker Compose

The simplest way to run Drift-ML is using Docker Compose:

```bash
cd docker
docker-compose up -d
```

This will start all services including Kafka, Redis, PostgreSQL, MLflow, and others.

## Running with Kubernetes

To deploy Drift-ML on Kubernetes:

1. Build the Docker image:
```bash
docker build -t drift-ml:latest .
```

2. Apply the Kubernetes configurations:
```bash
kubectl apply -f kubernetes/
```

3. Check the deployment status:
```bash
kubectl get all
```

## Security Integration

Drift-ML includes a comprehensive security module that provides:

- **Authentication**: JWT-based authentication with password hashing
- **Authorization**: Role-based access control
- **Encryption**: Data encryption for sensitive information
- **Audit Logging**: Comprehensive audit logging for all actions

# Run the security initialization and test script
./scripts/security_init.py

Security is integrated with:
- Model serving (complete)
- Feature store (complete)
- Data processing (complete)

## Getting Started

1. Default admin credentials:
   - Username: admin
   - Password: secure-password

2. Access the API at:
   - Local: http://localhost:8000
   - Kubernetes: http://[CLUSTER-IP]:8000

3. Example API calls:
```bash
# Get authentication token
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "secure-password"}'

# Use the token for authenticated requests
curl -X GET "http://localhost:8000/features" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

## Testing

Run the tests with:
```bash
pytest
```

## Documentation

See the `docs/` directory for detailed documentation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).