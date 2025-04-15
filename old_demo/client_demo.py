import time
import hashlib
import uuid

from model import Net, train
from dataset import load_data, load_data_non_iid
from collections import OrderedDict
import torch
import argparse

from fl_platform.src.utils.message_utils import ClientSimpleMessageHandler
import logging

model = Net()

parser = argparse.ArgumentParser(description='Client for federated learning platform.')
parser.add_argument("--cid", type=int, required=True, help="Client ID")
parser.add_argument("--num_cli", type=int, required=True, help="Number of clients")
args = parser.parse_args()

CID = args.cid
NUM_CLI = args.num_cli
TRAINLOADER, TESTLOADER = load_data(CID, NUM_CLI)
# TRAINLOADER, TESTLOADER = load_data_non_iid(CID, NUM_CLI)

if __name__ == '__main__':
    # Send server initiative to join
   
    # Get the MAC address
    mac = uuid.getnode()
    # Combine timestamp and MAC address
    data = f"{time.time()}_{mac}"
    # Compute the hash
    id = hashlib.sha256(data.encode()).hexdigest()
    id = id[:5]

    message_handler = ClientSimpleMessageHandler("localhost:29092", id, "global-models", "local-models", "client-logs")

    print(f"Client {id} started")
    message_handler.send_join_message()

    # Receive global params for this client

    while True:
        # print("Waiting for task...")

        try:
            responses = message_handler.consume_message(1000)
            if responses:
                for msg in responses:
                    params = msg.value.get('params')
                    
                    params_dict = zip(model.state_dict().keys(), params)
                    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                    model.load_state_dict(state_dict, strict=True)

                    logging.info(f"Client {id} training model...")

                    train_size = int(0.8 * len(TRAINLOADER.dataset))
                    val_size = len(TRAINLOADER.dataset) - train_size

                    train_subset, val_subset = torch.utils.data.random_split(TRAINLOADER.dataset, [train_size, val_size])
                    
                    trainloader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)
                    valloader = torch.utils.data.DataLoader(val_subset, batch_size=64, shuffle=False)
                    
                    results = train(model, trainloader, valloader, 10)
                    
                    # Send local params back to server

                    params = [val.cpu().numpy().tolist() for _, val in model.state_dict().items()]

                    message_handler.send_message({'client_id': id, 'params': params})

        except Exception as e:
            print(e)

        time.sleep(2)