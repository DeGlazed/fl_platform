from fl_platform.src.server import SimpleServer
from fl_platform.src.strategy.fed_fa import TaxiFedFA
import os

strategy = TaxiFedFA(k=6)

# for kubernetes
kafka_server = os.getenv('KAFKA_SERVER', 'localhost:9092')
localstack_server = os.getenv('LOCALSTACK_SERVER', 'http://localhost:4566')

print(f'Kafka server: {kafka_server}')
print(f'Localstack server: {localstack_server}')

server = SimpleServer(
    min_clients=6,
    strategy=strategy,

    kafka_server=kafka_server,
    
    client_logs_topic='client-logs',
    local_models_topic='local-models',
    global_models_topic='global-models',
    client_heartbeat_topic='client-heartbeat',
    server_heartbeat_topic='server-heartbeat',
    
    localstack_server= localstack_server,
    localstack_bucket='mybucket',

    # ca_certificate_file_path='kafka-certs/ca-cert.pem',
    # certificate_file_path='kafka-certs/client-cert.pem',
    # key_file_path='kafka-certs/client-key.pem'
)

server.start_server()