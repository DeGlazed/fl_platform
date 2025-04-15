from kafka.admin import KafkaAdminClient
from kafka import KafkaConsumer, KafkaProducer
import json
import time
import threading
from fl_platform.src.utils.client_manager import ClientManager, ClientState
from fl_platform.src.strategy.fed_fa import FedFA
from collections import OrderedDict
import torch
import argparse

from model import Net

from fl_platform.src.utils.message_utils import ServerSimpleMessageHandler

import hashlib
import uuid
import os
import boto3

local_models_consumer = KafkaConsumer(
    'local-models',
    bootstrap_servers='localhost:29092',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers='localhost:29092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

# Configure LocalStack
os.environ['AWS_ACCESS_KEY_ID'] = 'test'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'test'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
localstack_endpoint = 'http://localhost:4566'

# Create S3 client
s3 = boto3.client('s3', endpoint_url=localstack_endpoint)

# Create a bucket
bucket_name = 'your-bucket'

parser = argparse.ArgumentParser(description='Server for federated learning platform.')
parser.add_argument("--min_cli", type=int, required=True, help="Min Number Of Clients")
args = parser.parse_args()

MIN_CLIENTS = args.min_cli
client_manager = ClientManager()
strategy = FedFA(client_manager, MIN_CLIENTS)

model = Net()

# init_params = [val.cpu().numpy().tolist() for _, val in model.state_dict().items()]

init_params_filename = 'init_params.pth'
torch.save(model.state_dict(), init_params_filename)

def startClientHandler(message_handler) :
    while True:
        # Handle client connections
        results = message_handler.consume_client_logs_message(1000)
        if(results):
            for cli_id, msg in results :
                if cli_id:
                    print(f"Client {cli_id} connected")
                    client_manager.add_client(cli_id)

def startFLStrategy(message_handler, min_clinets_connected) :
    while True:
        #check for required number of clients connected
        ready_clinets_count = len(client_manager.get_all_ready_clients())
        if ready_clinets_count < min_clinets_connected:
            print(f"Waiting for {min_clinets_connected - ready_clinets_count} more clients to connect...")
            time.sleep(2)
        else :
            print(f"Minimum number of clients connected. Starting FL strategy...")
            break

    #send init params to min_clients_connected randomly selected clients

    print(f"Sending initial params to {min_clinets_connected} clients...")
    initial_population = strategy.get_initial_situation()

    s3.upload_file(init_params_filename, bucket_name, init_params_filename)
    for client_id in initial_population:
        print(f"Sending initial params to client {client_id}...")
        s3.upload_file(init_params_filename, bucket_name, init_params_filename)

        message_handler.send_message({'client_id': client_id, 'params': init_params_filename})
        client_manager.set_client_state(client_id, ClientState.BUSY)

    os.remove(init_params_filename)

    #Handle client responses
    while True:
        # print("Waiting for local params from clients...")

        results = message_handler.consume_local_model_message(1000)
        if results: 
            for client_id, msg in results:
                filename = msg.value.get('params')
                print(f"Received local params from client {client_id}")

                client_manager.set_client_state(client_id, ClientState.FINISHED)

                s3.download_file(bucket_name, filename, filename)
                model.load_state_dict(torch.load(filename))
                params = [val.cpu().numpy().tolist() for _, val in model.state_dict().items()]
                os.remove(filename)

                params_dict = zip(model.state_dict().keys(), params)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

                print(f"Processing params of {client_id}...")
                
                new_client_id, new_global_dict = strategy.agregate(state_dict)

                if(new_client_id):
                    print(f"Sending global params to client {new_client_id}...")

                    mac = uuid.getnode()
                    data = f"{time.time()}_{mac}"
                    hash = hashlib.sha256(data.encode()).hexdigest()
                    filename = hash + '.pt'
                    torch.save(new_global_dict, filename)
                    s3.upload_file(filename, bucket_name, filename)

                    message_handler.send_message({'client_id': new_client_id, 'params': filename})
                    client_manager.set_client_state(new_client_id, ClientState.BUSY)
                    os.remove(filename)

                print(f"Finished processing params of {client_id}")

                client_manager.set_client_state(client_id, ClientState.READY)      

        time.sleep(2)

if __name__ == '__main__':

    message_handler = ServerSimpleMessageHandler("localhost:29092", "global-models", "local-models", "client-logs")

    client_handler_thread = threading.Thread(target=startClientHandler, args=(message_handler,))
    client_handler_thread.daemon = True
    fl_strategy_thread = threading.Thread(target=startFLStrategy, args=(message_handler, MIN_CLIENTS,))
    fl_strategy_thread.daemon = True

    client_handler_thread.start()
    fl_strategy_thread.start()

    while True:
        time.sleep(1)