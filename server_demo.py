from kafka.admin import KafkaAdminClient
from kafka import KafkaConsumer, KafkaProducer
import json
import time
import threading
from fl_platform.src.utils.client_manager import ClientManager, ClientState
from fl_platform.src.strategy.fed_fa import FedFA
from collections import OrderedDict
import torch

from model import Net

admin_client = KafkaAdminClient(
    bootstrap_servers="localhost:29092", 
    client_id='test'
)

client_logs_consumer = KafkaConsumer(
    'client-logs',
    bootstrap_servers='localhost:29092',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

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

MIN_CLIENTS = 2
client_manager = ClientManager()
strategy = FedFA(client_manager, MIN_CLIENTS)

model = Net()
init_params = [val.cpu().numpy().tolist() for _, val in model.state_dict().items()]

def startClientHandler() :
    while True:
        # Handle client connections
        try:
            state = client_logs_consumer.poll(timeout_ms=1000)
            if state:
                for tp, states in state.items():
                    for msg in states:
                        client_id = msg.value.get('client_id')
                        if client_id:

                            client_manager.add_client(client_id)

        except Exception as e:
            print(e)

def startFLStrategy(min_clinets_connected) :
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

    for client_id in initial_population:
        print(f"Sending initial params to client {client_id}...")
        response = {'client_id': client_id, 'params': init_params}

        # response_size = len(json.dumps(response).encode('utf-8')) / (1024 * 1024)
        # print(f"Response size: {response_size:.2f} MB")

        producer.send('global-models', value=response)
        producer.flush()
        client_manager.set_client_state(client_id, ClientState.BUSY)

    #Handle client responses
    while True:
        print("Waiting for local params from clients...")
        try:
            state = local_models_consumer.poll(timeout_ms=1000)
            if state:
                for tp, states in state.items():
                    for msg in states:
                        client_id = msg.value.get('client_id')
                        if client_id:
                            params = msg.value.get('params')
                            print(f"Received local params from client {client_id}")

                            client_manager.set_client_state(client_id, ClientState.FINISHED)

                            params_dict = zip(model.state_dict().keys(), params)
                            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})

                            print(f"Processing params of {client_id}...")
                            
                            new_client_id, new_global_dict = strategy.agregate(state_dict)
                            if(new_client_id):
                                print(f"Sending global params to client {new_client_id}...")

                                params = [val.cpu().numpy().tolist() for _, val in new_global_dict.items()]

                                response = {'client_id': new_client_id, 'params': params}
                                producer.send('global-models', value=response)
                                producer.flush()
                                client_manager.set_client_state(new_client_id, ClientState.BUSY)

                            print(f"Finished processing params of {client_id}")

                            # # Send global params to random client in READY state
                            # random_client_ids = client_manager.sample_ready_clients(1)
                            # random_client_id = random_client_ids[0] if random_client_ids else None
                            # if(random_client_id):
                            #     print(f"Sending global params to client {random_client_id}...")
                            #     params = {'param1': 1, 'param2': 2}
                            #     response = {'client_id': random_client_id, 'params': params}
                            #     producer.send('global-models', value=response)
                            #     producer.flush()
                            #     client_manager.set_client_state(random_client_id, ClientState.BUSY)

                            client_manager.set_client_state(client_id, ClientState.READY)      

        except Exception as e:
            print(e)
        
        time.sleep(2)

if __name__ == '__main__':

    client_handler_thread = threading.Thread(target=startClientHandler)
    fl_strategy_thread = threading.Thread(target=startFLStrategy, args=(MIN_CLIENTS,))

    client_handler_thread.start()
    fl_strategy_thread.start()

    client_handler_thread.join()
    fl_strategy_thread.join()