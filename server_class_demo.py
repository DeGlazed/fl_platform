from fl_platform.src.server import SimpleServer
from fl_platform.src.strategy.fed_fa import FedFA

strategy = FedFA(k=2)  # Example strategy with k=5

server = SimpleServer(
    min_clients=2,
    strategy=strategy,
    
    kafka_server='localhost:29092',
    client_logs_topic='client-logs',
    local_models_topic='local-models',
    global_models_topic='global-models',
    client_heartbeat_topic='client-heartbeat',
    server_heartbeat_topic='server-heartbeat',
    localstack_server='http://localhost:4566',
    localstack_bucket='mybucket',
)

server.start_server()