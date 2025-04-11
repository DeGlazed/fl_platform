from fl_platform.src.client import SimpleClient
from model import MiniNet, train
from dataset import load_data
import argparse
import time
import torch

parser = argparse.ArgumentParser(description='FL Client')
parser.add_argument('--c', type=int, default=0, help='Partition ID for the client')
parser.add_argument('--p', type=int, default=1, help='Number of partitions for the client')

args = parser.parse_args()
partition_id = args.c
num_partitions = args.p

trainloader, testloader = load_data(partition_id, num_partitions)

model = MiniNet()

client = SimpleClient(
    model=model,
    kafka_server='localhost:29092',
    client_logs_topic='client-logs',
    local_models_topic='local-models',
    global_models_topic='global-models',
    client_heartbeat_topic='client-heartbeat',
    server_heartbeat_topic='server-heartbeat',
    localstack_server='http://localhost:4566',
    localstack_bucket='mybucket',
)

while True:
    new_model = client.get_new_task()
    if new_model:
        print("Recv new model")
        
        print("Start training")
        results = train(model, trainloader, testloader, epochs=10)
        print("Finished training with results:", results)

        client.publish_updated_model(new_model, results)
        print("Sent model")

    time.sleep(3)

# client.close()