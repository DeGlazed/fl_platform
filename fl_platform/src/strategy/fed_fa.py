from .strategy import AbstractStrategy
from typing import List
import numpy as np
from collections import OrderedDict
import logging

#Source : FEDFA: A fully asynchronous training paradigm for federated learning. (n.d.). https://arxiv.org/html/2404.11015v2
class FedFA(AbstractStrategy) :
    def __init__(self,
                 k: int
                 ) :
        self.k = k
        self.queue = []
        self.first_round = True

    def compute_new_global_model(self) -> OrderedDict:
        mean_params = OrderedDict()
        for key in self.queue[0].keys():
            mean_params[key] = np.mean([state_dict[key] for state_dict in self.queue], axis=0)
        return mean_params
    
    def get_number_of_initial_client_samples(self) -> int:
        return self.k
    
    def aggregate(self, state_dict: OrderedDict, training_info : dict) -> List:
       
        if training_info:
            logging.debug(f"Training info: {training_info}")

        self.queue.append(state_dict)
        
        if len(self.queue) < self.k:
            logging.debug("Queue not full yet")
            return None, None
        
        if len(self.queue) > self.k :
            self.queue.pop(0)
        
        logging.debug("Queue full, computing new global model")
        new_global_model = self.compute_new_global_model()
        
        if self.first_round:
            self.first_round = False
            return self.k, new_global_model
        
        return 1, new_global_model