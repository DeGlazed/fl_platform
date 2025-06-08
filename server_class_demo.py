from fl_platform.src.server import SimpleServer
from fl_platform.src.strategy.fed_fa import NaiveFedFA, SampleSizeAwareFedFA, TimestampSizeAwareFedFA, DataQualityAwareFedFA

# strategy = NaiveFedFA(k=3)
# strategy = SampleSizeAwareFedFA(k=3)
# strategy = TimestampSizeAwareFedFA(k=3)
strategy = DataQualityAwareFedFA(k=3)

# for docker
# kafka_server='localhost:9092', #PLAINTEXT
kafka_server='localhost:9095', #SSL
localstack_server='http://localhost:4566'

## for kubernetes
# kafka_server='localhost:30095', #SSL
# localstack_server='http://localhost:30566'

server = SimpleServer(
    min_clients=3,
    strategy=strategy,

    kafka_server=kafka_server,
    
    client_logs_topic='client-logs',
    local_models_topic='local-models',
    global_models_topic='global-models',
    client_heartbeat_topic='client-heartbeat',
    server_heartbeat_topic='server-heartbeat',
    
    localstack_server= localstack_server,
    localstack_bucket='mybucket',

    ca_certificate_file_path='kafka-certs/ca-cert.pem',
    certificate_file_path='kafka-certs/client-cert.pem',
    key_file_path='kafka-certs/client-key.pem'
)

server.start_server()