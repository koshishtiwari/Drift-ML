#!/bin/bash

set -e

# Configuration
NAMESPACE=${NAMESPACE:-drift-ml}
REGISTRY=${REGISTRY:-docker.io/driftml}
TAG=${TAG:-latest}
MODE=${MODE:-all}

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Deploying Drift-ML platform (Mode: ${MODE})${NC}"

# Check kubectl access
if ! kubectl cluster-info > /dev/null 2>&1; then
    echo -e "${RED}Error: Cannot connect to Kubernetes cluster${NC}"
    exit 1
fi

# Create namespace if it doesn't exist
if ! kubectl get namespace ${NAMESPACE} > /dev/null 2>&1; then
    echo -e "${YELLOW}Creating namespace: ${NAMESPACE}${NC}"
    kubectl create namespace ${NAMESPACE}
fi

# Apply resource quotas
echo -e "${GREEN}Applying resource quotas...${NC}"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ResourceQuota
metadata:
  name: drift-ml-quota
  namespace: ${NAMESPACE}
spec:
  hard:
    requests.cpu: "8"
    requests.memory: 16Gi
    limits.cpu: "16"
    limits.memory: 32Gi
    pods: "30"
EOF

# Deploy secrets
echo -e "${GREEN}Creating secrets...${NC}"
kubectl create secret generic drift-ml-secrets \
    --namespace=${NAMESPACE} \
    --from-literal=jwt-secret=$(openssl rand -base64 32) \
    --from-literal=db-password=$(openssl rand -base64 16) \
    --dry-run=client -o yaml | kubectl apply -f -

# Apply storage
echo -e "${GREEN}Setting up persistent storage...${NC}"
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-pvc
  namespace: ${NAMESPACE}
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
EOF

# Process template files with environment variables
echo -e "${GREEN}Processing template files...${NC}"
for template in config/kubernetes/*.yaml; do
    if [ -f "$template" ]; then
        output_file="/tmp/$(basename ${template})"
        envsubst < ${template} > ${output_file}
        kubectl apply -f ${output_file} --namespace=${NAMESPACE}
    fi
done

# Deploy monitoring stack if selected
if [[ "$MODE" == "all" || "$MODE" == "monitoring" ]]; then
    echo -e "${GREEN}Deploying monitoring stack...${NC}"
    kubectl apply -f config/kubernetes/prometheus-operator.yaml --namespace=${NAMESPACE}
    kubectl apply -f config/kubernetes/grafana.yaml --namespace=${NAMESPACE}
    kubectl apply -f config/kubernetes/prometheus.yaml --namespace=${NAMESPACE}
    
    # Wait for Prometheus CRDs to be ready
    echo -e "${YELLOW}Waiting for Prometheus CRDs to be available...${NC}"
    kubectl wait --for=condition=established --timeout=60s \
      crd/prometheusrules.monitoring.coreos.com \
      crd/servicemonitors.monitoring.coreos.com \
      crd/podmonitors.monitoring.coreos.com
    
    # Apply monitoring resources
    kubectl apply -f config/kubernetes/monitoring/ --namespace=${NAMESPACE}
fi

# Deploy model server
echo -e "${GREEN}Deploying model server...${NC}"
kubectl apply -f config/kubernetes/model-server-deployment.yaml --namespace=${NAMESPACE}
kubectl apply -f config/kubernetes/model-server-service.yaml --namespace=${NAMESPACE}
kubectl apply -f config/kubernetes/model-server-hpa.yaml --namespace=${NAMESPACE}

# Create ServiceMonitor for Prometheus to scrape metrics
cat <<EOF | kubectl apply -f -
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: model-server-monitor
  namespace: ${NAMESPACE}
  labels:
    app: drift-ml
spec:
  selector:
    matchLabels:
      app: drift-ml
      component: model-server
  endpoints:
  - port: http
    path: /metrics
    interval: 15s
EOF

# Wait for deployments to be ready
echo -e "${YELLOW}Waiting for deployments to be ready...${NC}"
kubectl rollout status deployment/model-server --namespace=${NAMESPACE} --timeout=300s

# Show access information
echo -e "${GREEN}=============================${NC}"
echo -e "${GREEN}Drift-ML Platform Deployed!${NC}"
echo -e "${GREEN}=============================${NC}"
echo ""
echo -e "Model Server API: http://$(kubectl get service model-server --namespace=${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8080"
echo -e "Grafana Dashboard: http://$(kubectl get service grafana --namespace=${NAMESPACE} -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):3000"
echo ""
echo -e "${YELLOW}Default Grafana credentials: admin / admin${NC}"
echo -e "${YELLOW}Remember to change the default password on first login${NC}"
