from fl_platform.src.server import SimpleServer
from fl_platform.src.strategy.fed_fa import FedFA

strategy = FedFA(k=2)  # Example strategy with k=5

server = SimpleServer(
    min_clients=2,
    strategy=strategy,

    # kafka_server='localhost:9092', #PLAINTEXT
    kafka_server='localhost:9095', #SSL
    
    client_logs_topic='client-logs',
    local_models_topic='local-models',
    global_models_topic='global-models',
    client_heartbeat_topic='client-heartbeat',
    server_heartbeat_topic='server-heartbeat',
    localstack_server='http://localhost:4566',
    localstack_bucket='mybucket',

    ca_certificate_file_path='kafka-certs/ca-cert.pem',
    certificate_file_path='kafka-certs/client-cert.pem',
    key_file_path='kafka-certs/client-key.pem'
)

server.start_server()