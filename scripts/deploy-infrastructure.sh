#!/bin/bash
set -e

# Function to echo steps with formatting
log() {
  echo -e "\n\033[1;34m>> $1\033[0m"
}

# Function to check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Function to create secrets
create_secret() {
  local namespace=$1
  local name=$2
  local from_literal=$3
  
  echo "Creating secret $name in namespace $namespace"
  kubectl create secret generic $name \
    --namespace $namespace \
    --from-literal=$from_literal \
    --dry-run=client -o yaml | kubectl apply -f -
}

# Pre-flight checks
log "Checking prerequisites"
for cmd in kubectl helm aws; do
  if ! command_exists $cmd; then
    echo "Error: $cmd is not installed or not in PATH"
    exit 1
  fi
done

# Create namespaces
log "Creating Drift-ML namespaces"
kubectl apply -f infrastructure/kubernetes/namespaces.yaml

# Deploy cert-manager for TLS certificates
log "Installing cert-manager"
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Wait for cert-manager to be ready before proceeding
log "Waiting for cert-manager to be ready"
kubectl wait --for=condition=available --timeout=180s deployment/cert-manager-webhook -n cert-manager
kubectl wait --for=condition=available --timeout=180s deployment/cert-manager-cainjector -n cert-manager
kubectl wait --for=condition=available --timeout=180s deployment/cert-manager -n cert-manager

# Create certificate issuer and certs
log "Configuring TLS certificates"
kubectl apply -f infrastructure/kubernetes/security/certificate-issuer.yaml
sleep 5  # Brief pause to let the CRDs register

# Create required secrets
log "Creating secrets for services"
create_secret "drift-ml-data" "postgresql-credentials" "admin-password=postgres_admin_pw,user-password=drift_ml_user_pw,replication-password=repl_pw"
create_secret "drift-ml-data" "kafka-jaas-config" "kafka_jaas.conf=KafkaServer { org.apache.kafka.common.security.scram.ScramLoginModule required username=\"admin\" password=\"admin-secret\"; };"
create_secret "drift-ml-data" "minio-credentials" "access-key=drift-ml-user-access-key,secret-key=drift-ml-user-secret-key"
create_secret "drift-ml-data" "redis-credentials" "redis-password=redis_complex_password"
create_secret "drift-ml-data" "schema-registry-credentials" "basic.properties=schema.registry.auth.user.info=admin:admin-secret,reader:reader-secret,writer:writer-secret"

# Add Helm repositories
log "Adding Helm repositories"
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add open-telemetry https://open-telemetry.github.io/opentelemetry-helm-charts
helm repo update

# Deploy custom GPU scheduler first to ensure proper scheduling
log "Deploying custom GPU scheduler"
kubectl apply -f infrastructure/kubernetes/custom-scheduler/gpu-scheduler.yaml
kubectl wait --for=condition=available --timeout=60s deployment/gpu-scheduler -n kube-system

# Deploy PostgreSQL
log "Deploying PostgreSQL"
helm install postgresql bitnami/postgresql \
  --namespace drift-ml-data \
  --values infrastructure/helm/postgresql/values.yaml

# Deploy MinIO (S3-compatible storage)
log "Deploying MinIO for object storage"
helm install minio bitnami/minio \
  --namespace drift-ml-data \
  --values infrastructure/helm/minio/values.yaml

# Deploy Redis
log "Deploying Redis cluster"
helm install redis bitnami/redis \
  --namespace drift-ml-data \
  --values infrastructure/helm/redis/values.yaml

# Deploy Kafka
log "Deploying Kafka in KRaft mode"
helm install kafka bitnami/kafka \
  --namespace drift-ml-data \
  --values infrastructure/helm/kafka/values.yaml

# Wait for critical data services to be ready before monitoring
log "Waiting for core data services to be ready"
kubectl wait --for=condition=available --timeout=300s deployment -l app.kubernetes.io/instance=postgresql -n drift-ml-data || echo "Warning: PostgreSQL not fully ready, but continuing..."
kubectl wait --for=condition=available --timeout=300s deployment -l app.kubernetes.io/instance=redis -n drift-ml-data || echo "Warning: Redis not fully ready, but continuing..."
kubectl wait --for=condition=available --timeout=300s deployment -l app.kubernetes.io/instance=minio -n drift-ml-data || echo "Warning: MinIO not fully ready, but continuing..."

# Deploy monitoring stack
log "Deploying Prometheus monitoring stack"
helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace drift-ml-monitoring \
  --values infrastructure/helm/monitoring/prometheus-values.yaml

log "Deploying OpenTelemetry Collector"
helm install otel-collector open-telemetry/opentelemetry-collector \
  --namespace drift-ml-monitoring \
  --values infrastructure/helm/monitoring/otel-collector-values.yaml

# Setup Kafka users and ACLs after Kafka is ready
log "Waiting for Kafka to be ready"
kubectl wait --for=condition=available --timeout=300s deployment -l app.kubernetes.io/instance=kafka -n drift-ml-data || echo "Warning: Kafka not fully ready, but continuing with user setup..."

log "Setting up Kafka authentication"
kubectl apply -f infrastructure/kubernetes/security/kafka-auth.yaml

log "Infrastructure deployment complete! Phase 0 has been implemented."
echo "You can verify the deployments with: kubectl get pods --all-namespaces | grep drift-ml"
