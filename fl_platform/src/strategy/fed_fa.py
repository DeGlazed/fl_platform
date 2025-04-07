from .strategy import AbstractStrategy
from typing import List
import numpy as np

#Source : FEDFA: A fully asynchronous training paradigm for federated learning. (n.d.). https://arxiv.org/html/2404.11015v2
class FedFA(AbstractStrategy) :
    def __init__(self,
                 k: int
                 ) :
        self.k = k
        self.queue = []

    def compute_new_global_model(self) -> List:
        mean_params = []
        for param_set in zip(*self.queue):
            mean_param = np.mean(param_set, axis=0).tolist()
            mean_params.append(mean_param)
        return mean_params
    
    def get_number_of_initial_client_samples(self) -> int:
        return self.k
    
    def agregate(self, local_model: List) -> List:
       
        self.queue.append(local_model)
        
        if len(self.queue) < self.k:
            print("Queue not full yet")
            return None, None
        
        if len(self.queue) > self.k :
            self.queue.pop(0)
        
        print("Queue full, computing new global model")
        new_global_model = self.compute_new_global_model()
        
        # save_dir = "model_states"
        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)
        # index = len(os.listdir(save_dir))
        # file_path = os.path.join(save_dir, f"model_{index}.pth")
        # torch.save(new_global_model, file_path)
        
        return new_global_model