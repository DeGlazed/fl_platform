from fl_platform.src.utils.client_manager import ClientManager
from typing import List, OrderedDict, Tuple
import os
import torch

#Source : FEDFA: A fully asynchronous training paradigm for federated learning. (n.d.). https://arxiv.org/html/2404.11015v2
class FedFA() :
    def __init__(self, 
                 client_manager: ClientManager, 
                 k: int
                 ) :
        self.k = k
        self.client_manager = client_manager
        self.queue = []

    def compute_new_global_model(self) -> OrderedDict:
        sum_model = OrderedDict()
        for model in self.queue:
            for key, value in model.items():
                if key in sum_model:
                    sum_model[key] += value
                else:
                    sum_model[key] = value
        
        avg_model = OrderedDict()
        for key, value in sum_model.items():
            avg_model[key] = value / self.k
        
        return avg_model
    
    def get_initial_situation(self) -> List[str]:
        return self.client_manager.sample_ready_clients(self.k)
    
    def agregate(self, local_model: OrderedDict) -> Tuple[str, OrderedDict]:
       
        self.queue.append(local_model)
        
        if len(self.queue) < self.k:
            print("Queue not full yet")
            return None, None
        
        if len(self.queue) > self.k :
            self.queue.pop(0)
        
        print("Queue full, computing new global model")
        new_global_model = self.compute_new_global_model()
        
        save_dir = "model_states"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        index = len(os.listdir(save_dir))
        file_path = os.path.join(save_dir, f"model_{index}.pth")

        torch.save(new_global_model, file_path)

        client_id = self.client_manager.sample_ready_clients(1)[0]
        return client_id, new_global_model