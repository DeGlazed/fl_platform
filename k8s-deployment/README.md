# For SSL certificates, add them to secrets
kubectl create secret generic kafka-tls-secret --from-file=kafka.keystore.jks=.\kafka-certs\kafka.keystore.jks --from-file=kafka.truststore.jks=.\kafka-certs\kafka.truststore.jks

# Portforwarding
kubectl port-forward svc/kafka 30095:9095 -n default
kubectl port-forward svc/localstack 30566:4566 -n default

# Access web service outside of cluster
minikube service kafka-ui --url

# Create configmaps from files scripts
kubectl create configmap prometheus-config --from-file=prometheus.yml=.\prometheus\prometheus.yml --namespace=default --dry-run=client -o yaml > prometheus-configmap.yaml

kubectl create configmap grafana-datasources --from-file=./grafana/datasources --namespace=default --dry-run=client -o yaml > grafana-datasources-configmap.yaml

kubectl create configmap grafana-dashboards --from-file=./grafana/dashboards --namespace=default --dry-run=client -o yaml > grafana-dashboards-configmap.yaml