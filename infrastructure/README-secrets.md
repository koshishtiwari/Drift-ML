# Secrets Management for Drift-ML

This document provides guidance on setting up and managing secrets for the Drift-ML platform.

## HashiCorp Vault Integration

We use HashiCorp Vault for secure secrets management in the Drift-ML platform. The setup includes:

1. Vault deployed in HA mode with Raft storage
2. Kubernetes authentication for services
3. Dynamic secrets for databases and messaging systems
4. Secret rotation policies

### Initial Setup

The initial Vault setup is handled by the `vault-integration.yaml` manifest, which:

1. Creates a dedicated namespace for Vault
2. Deploys Vault in HA mode with 3 replicas
3. Configures TLS using cert-manager
4. Sets up Kubernetes authentication
5. Creates initial policies and roles

### Accessing Vault

After deployment, retrieve the initial root token and unseal keys:

```bash
kubectl get secret vault-init -n vault -o jsonpath='{.data.init}' | base64 -d
```

Store these securely as they're needed to unseal Vault after restarts.

### Creating Service Credentials

To create service-specific credentials:

1. Create a Kubernetes ServiceAccount for your application
2. Annotate it with the appropriate Vault role
3. Mount the Vault Agent sidecar in your deployment

Example service account:

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: feature-registry
  namespace: drift-ml-data
  annotations:
    vault.hashicorp.io/agent-inject: "true"
    vault.hashicorp.io/agent-inject-secret-db-creds: "drift-ml-db/creds/feature-registry"
    vault.hashicorp.io/role: "drift-ml-role"
```

### Rotation Policies

Credentials are automatically rotated based on the following schedule:

- Database credentials: Every 24 hours
- Kafka credentials: Every 7 days
- API tokens: Every 30 days

## Kubernetes Secrets (Fallback)

For environments without Vault, we use Kubernetes secrets with these best practices:

1. Never commit secrets to Git
2. Use environment-specific secret files
3. Consider using SealedSecrets or ExternalSecrets for GitOps flows

## Local Development

For local development, use a `.env.local` file that is git-ignored:

```bash
# Create a local environment file
cp .env.example .env.local

# Edit with your credentials
nano .env.local
```

## Production Considerations

In production environments:
1. Enable auto-unseal with a cloud KMS
2. Set up audit logging
3. Implement secret access monitoring
4. Regular credential rotation

## Security Compliance

Our secrets management approach complies with:
- SOC 2 requirements
- GDPR requirements for data protection
- Industry best practices for credential management
