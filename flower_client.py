from centralized import load_data, train
import flwr as fl
from fl_platform.src.models.model import SimpleLSTM

from collections import OrderedDict
import torch
import argparse
import time

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

argparser = argparse.ArgumentParser(description='FL Client')
argparser.add_argument('--c', type=int, default=0, help='Client ID')
argparser.add_argument('--p', type=int, default=1, help='Number of total clients')

args = argparser.parse_args()
partition_id = args.c
num_partitions = args.p + 1

def set_parameters(model, params):
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.tensor(v) for k,v in params_dict})
    model.load_state_dict(state_dict, strict = True)
    return model

dataloader, dataset = load_data(partition_id, num_partitions)
input_size = 5
hidden_size = 64
num_layers = 1
num_classes = len(dataset.label_mapping)
net = SimpleLSTM(input_size, hidden_size, num_layers, num_classes).to(DEVICE) 



class FlowerClient(fl.client.NumPyClient) :
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]
    
    def fit(self, parameters, config):
        set_parameters(net, parameters)

        with open("c" + str(partition_id) + "_flower_work.txt", "a") as file:
            file.write(f"{len(dataloader.dataset)}\n")

        train(net, dataloader, num_epochs=5, lr=1e-3)
        return self.get_parameters({}), len(dataloader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        torch.save(net.state_dict(), f"{time.time()}_model.pt")
        return 0.0, len(dataloader.dataset), {"accuracy": 0.0}

if __name__ == "__main__" :
    fl.client.start_numpy_client(
        server_address="127.0.0.1:8081",
        client=FlowerClient().to_client(),
    )