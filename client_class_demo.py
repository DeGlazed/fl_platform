from fl_platform.src.client import SimpleClient
from fl_platform.src.models.model import SimpleLSTM
import argparse
import torch
import time

from centralized import load_data, train


parser = argparse.ArgumentParser(description='FL Client')
parser.add_argument('--c', type=int, default=0, help='Client ID')
parser.add_argument('--p', type=int, default=1, help='Number of total clients')

args = parser.parse_args()
partition_id = args.c
num_partitions = args.p + 1

dataloader, dataset = load_data(partition_id, num_partitions)

input_size = 5
hidden_size = 64
num_layers = 1
num_classes = len(dataset.label_mapping)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleLSTM(input_size, hidden_size, num_layers, num_classes)

# for docker
# kafka_server='localhost:9092', #PLAINTEXT
kafka_server='localhost:9095', #SSL
localstack_server='http://localhost:4566'

## for kubernetes
# kafka_server='localhost:30095', #SSL
# localstack_server='http://localhost:30566'

client = SimpleClient(
    model=model,
    kafka_server=kafka_server,
    
    client_logs_topic='client-logs',
    local_models_topic='local-models',
    global_models_topic='global-models',
    client_heartbeat_topic='client-heartbeat',
    server_heartbeat_topic='server-heartbeat',
    localstack_server=localstack_server,
    localstack_bucket='mybucket',

    ca_certificate_file_path='kafka-certs/ca-cert.pem',
    certificate_file_path='kafka-certs/client-cert.pem',
    key_file_path='kafka-certs/client-key.pem'
)

model.to(device)
start_time = time.time()

while time.time() - start_time < 1800: 
    new_model = client.get_new_task()
    if new_model:
        print("Recv new model")
        print("Start training")
        
        # with open(client.cid + "_work.txt", "a") as file:
        #     file.write(f"{len(dataloader.dataset)}\n")

        train(new_model, dataloader, num_epochs=5, lr=1e-3)

        client.publish_updated_model(new_model)
        print("Sent model")

client.close()
print("Client closed")
