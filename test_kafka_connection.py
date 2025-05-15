import ssl
from kafka.admin import KafkaAdminClient, NewTopic

kafka_server='localhost:30095'
#need to port forward the kafka server to localhost

admin_client = KafkaAdminClient(
    bootstrap_servers=kafka_server,
    client_id='admin',
    security_protocol='SSL',
    ssl_cafile='kafka-certs/ca-cert.pem',
    ssl_certfile='kafka-certs/client-cert.pem',
    ssl_keyfile='kafka-certs/client-key.pem'
)

print(admin_client.list_topics())