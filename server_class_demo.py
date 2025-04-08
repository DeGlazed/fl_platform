from fl_platform.src.server import SimpleServer
from fl_platform.src.strategy.fed_fa import FedFA
from model import Net

strategy = FedFA(k=1)  # Example strategy with k=5

server = SimpleServer(
    min_clients=1,
    strategy=strategy,
    
    kafka_server='localhost:29092',
    client_logs_topic='client-logs',
    local_models_topic='local-models',
    global_models_topic='global-models',
    localstack_server='http://localhost:4566',
    localstack_bucket='mybucket',
)

server.start_server()