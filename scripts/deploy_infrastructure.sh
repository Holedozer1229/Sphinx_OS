#!/bin/bash
set -e

echo "🚀 Deploying Sphinx_OS Infrastructure to Production"
echo "=================================================="

# Configuration
ENVIRONMENT=${ENVIRONMENT:-mainnet}
NAMESPACE="sphinxos-prod"
DOMAIN=${DOMAIN:-"sphinxos.io"}

# Check requirements
command -v kubectl >/dev/null 2>&1 || { echo "kubectl required"; exit 1; }
command -v helm >/dev/null 2>&1 || { echo "helm required"; exit 1; }

echo "✅ Prerequisites checked"

# Create namespace
echo "📦 Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Deploy secrets
echo "🔐 Creating secrets..."
kubectl create secret generic sphinxos-secrets \
  --from-literal=jwt-secret="${JWT_SECRET:?JWT_SECRET not set}" \
  --from-literal=database-url="${DATABASE_URL:?DATABASE_URL not set}" \
  --from-literal=redis-url="${REDIS_URL:?REDIS_URL not set}" \
  --from-literal=sentry-dsn="${SENTRY_DSN:-}" \
  --namespace=$NAMESPACE \
  --dry-run=client -o yaml | kubectl apply -f -

# Deploy PostgreSQL
echo "📦 Deploying PostgreSQL..."
helm repo add bitnami https://charts.bitnami.com/bitnami || true
helm repo update
helm upgrade --install sphinxos-postgres bitnami/postgresql \
  --namespace=$NAMESPACE \
  --set auth.postgresPassword="${POSTGRES_PASSWORD:?POSTGRES_PASSWORD not set}" \
  --set persistence.size=100Gi \
  --set primary.resources.requests.memory=2Gi \
  --set primary.resources.requests.cpu=1000m \
  --wait

# Deploy Redis
echo "📦 Deploying Redis..."
helm upgrade --install sphinxos-redis bitnami/redis \
  --namespace=$NAMESPACE \
  --set auth.password="${REDIS_PASSWORD:?REDIS_PASSWORD not set}" \
  --set master.persistence.size=10Gi \
  --wait

# Deploy Sphinx_OS application
echo "🧠 Deploying Sphinx_OS nodes..."
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sphinxos-node
  namespace: $NAMESPACE
  labels:
    app: sphinxos-node
spec:
  replicas: 10
  selector:
    matchLabels:
      app: sphinxos-node
  template:
    metadata:
      labels:
        app: sphinxos-node
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8001"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: sphinxos
        image: ${DOCKER_IMAGE:-sphinxos:latest}
        ports:
        - containerPort: 8000
          name: http
        - containerPort: 8001
          name: metrics
        env:
        - name: SPHINXOS_ENV
          value: "$ENVIRONMENT"
        - name: JWT_SECRET
          valueFrom:
            secretKeyRef:
              name: sphinxos-secrets
              key: jwt-secret
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: sphinxos-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: sphinxos-secrets
              key: redis-url
        - name: SENTRY_DSN
          valueFrom:
            secretKeyRef:
              name: sphinxos-secrets
              key: sentry-dsn
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: sphinxos-node
  namespace: $NAMESPACE
spec:
  selector:
    app: sphinxos-node
  ports:
  - name: http
    port: 80
    targetPort: 8000
  - name: metrics
    port: 8001
    targetPort: 8001
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sphinxos-api
  namespace: $NAMESPACE
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/limit-rps: "10"
spec:
  tls:
  - hosts:
    - api.$DOMAIN
    secretName: sphinxos-api-tls
  rules:
  - host: api.$DOMAIN
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sphinxos-node
            port:
              number: 80
EOF

echo "Waiting for deployment to be ready..."
kubectl rollout status deployment/sphinxos-node -n $NAMESPACE

# Deploy monitoring stack
echo "📊 Deploying monitoring stack..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts || true
helm repo update
helm upgrade --install sphinxos-monitoring prometheus-community/kube-prometheus-stack \
  --namespace=$NAMESPACE \
  --set grafana.ingress.enabled=true \
  --set grafana.ingress.hosts[0]="grafana.$DOMAIN" \
  --set grafana.ingress.tls[0].hosts[0]="grafana.$DOMAIN" \
  --set grafana.ingress.tls[0].secretName=sphinxos-grafana-tls \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --wait

# Deploy horizontal pod autoscaler
echo "📈 Configuring auto-scaling..."
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sphinxos-node-hpa
  namespace: $NAMESPACE
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sphinxos-node
  minReplicas: 10
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF

echo ""
echo "✅ Deployment complete!"
echo ""
echo "🌐 URLs:"
echo "  API: https://api.$DOMAIN"
echo "  Grafana: https://grafana.$DOMAIN"
echo ""
echo "📊 Check status:"
echo "  kubectl get pods -n $NAMESPACE"
echo "  kubectl logs -n $NAMESPACE -l app=sphinxos-node"
echo ""
echo "🔍 View metrics:"
echo "  kubectl port-forward -n $NAMESPACE svc/sphinxos-monitoring-kube-prom-prometheus 9090:9090"
echo ""
