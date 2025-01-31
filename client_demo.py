from kafka.admin import KafkaAdminClient, NewTopic
from kafka import KafkaConsumer, KafkaProducer
import json
import time
import hashlib
import uuid

from model import Net, train
from dataset import load_data
from collections import OrderedDict
import torch

global_models_consumer = KafkaConsumer(
    'global-models',
    bootstrap_servers='localhost:29092',
    auto_offset_reset='latest',
    enable_auto_commit=True,
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

producer = KafkaProducer(
    bootstrap_servers='localhost:29092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

model = Net()
TRAINLOADER, VALLOADER, TESTLOADER = load_data(0, 2)

if __name__ == '__main__':
    # Send server initiative to join
   
    # Get the MAC address
    mac = uuid.getnode()
    # Combine timestamp and MAC address
    data = f"{time.time()}_{mac}"
    # Compute the hash
    id = hashlib.sha256(data.encode()).hexdigest()
    id = id[:5]

    print(f"I am {id}")
    print("Sending state...")
    response = {'state': 'CONNECT', 'client_id': id}
    producer.send('client-logs', value=response)
    producer.flush()

    # Receive global params for this client

    while True:
        print("Waiting for task...")

        try:
            state = global_models_consumer.poll(timeout_ms=1000)
            if state:
                for tp, states in state.items():
                    for msg in states:
                        client_id = msg.value.get('client_id')
                        if(id == client_id):
                            params = msg.value.get('params')

                            params_dict = zip(model.state_dict().keys(), params)
                            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                            model.load_state_dict(state_dict, strict=True)

                            print("Training model...")
                            
                            train(model, TRAINLOADER, VALLOADER, 10)

                            print("Model trained, sending local params back to server...")
                            # Send local params back to server

                            params = [val.cpu().numpy().tolist() for _, val in model.state_dict().items()]

                            response = {'client_id': id, 'params': params}
                            producer.send('local-models', value=response)
                            producer.flush()

        except Exception as e:
            print(e)

        time.sleep(2)