from .strategy import AbstractStrategy
from typing import List
import numpy as np
from collections import OrderedDict
import logging
import pandas as pd

#Inspiration : FEDFA: A fully asynchronous training paradigm for federated learning. (n.d.). https://arxiv.org/html/2404.11015v2
class NaiveFedFA(AbstractStrategy) :
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
       
        logging.info(f"Training info: {training_info}")

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
    
class SampleSizeAwareFedFA(AbstractStrategy) :
    def __init__(self,
                 k: int
                 ) :
        self.k = k
        self.info_queue = []
        self.params_queue = []
        self.first_round = True

    def compute_new_global_model(self) -> OrderedDict:
        mean_params = OrderedDict()
        total_samples = sum([info['num_samples'] for info in self.info_queue])
        for key in self.params_queue[0].keys():
            weighted_sum = np.zeros_like(self.params_queue[0][key])
            for i, state_dict in enumerate(self.params_queue):
                weight = self.info_queue[i]['num_samples'] / total_samples
                weighted_sum += weight * state_dict[key]
            mean_params[key] = weighted_sum
        return mean_params
    
    def get_number_of_initial_client_samples(self) -> int:
        return self.k
    
    def aggregate(self, state_dict: OrderedDict, training_info : dict) -> List:
       
        logging.info(f"Training info: {training_info}")

        self.info_queue.append(training_info)
        self.params_queue.append(state_dict)

        if len(self.params_queue) < self.k:
            logging.debug("Queue not full yet")
            return None, None

        if len(self.params_queue) > self.k :
            self.info_queue.pop(0)
            self.params_queue.pop(0)

        logging.debug("Queue full, computing new global model")
        new_global_model = self.compute_new_global_model()
        
        if self.first_round:
            self.first_round = False
            return self.k, new_global_model
        
        return 1, new_global_model

class TimestampSizeAwareFedFA(AbstractStrategy):
    def __init__(self, k: int, decay_factor: float = 0.9):
        self.k = k
        self.decay_factor = decay_factor
        self.info_queue = []
        self.params_queue = []
        self.first_round = True

    def compute_new_global_model(self) -> OrderedDict:
        mean_params = OrderedDict()

        current_time = max([float(info['timestamp']) for info in self.info_queue])
        
        total_weighted_samples = 0
        weights = []
        for info in self.info_queue:
            time_weight = self.decay_factor ** (current_time - float(info['timestamp']))
            sample_weight = info['num_samples']
            combined_weight = time_weight * sample_weight
            weights.append(combined_weight)
            total_weighted_samples += combined_weight
        
        # Normalize weights
        weights = [w / total_weighted_samples for w in weights]
        
        for key in self.params_queue[0].keys():
            weighted_sum = np.zeros_like(self.params_queue[0][key])
            for i, state_dict in enumerate(self.params_queue):
                weighted_sum += weights[i] * state_dict[key]
            mean_params[key] = weighted_sum
        
        return mean_params
    
    def get_number_of_initial_client_samples(self) -> int:
        return self.k
    
    def aggregate(self, state_dict: OrderedDict, training_info: dict) -> List:
        logging.info(f"Training info: {training_info}")

        self.info_queue.append(training_info)
        self.params_queue.append(state_dict)

        if len(self.params_queue) < self.k:
            logging.debug("Queue not full yet")
            return None, None

        if len(self.params_queue) > self.k:
            self.info_queue.pop(0)
            self.params_queue.pop(0)

        logging.debug("Queue full, computing new global model")
        new_global_model = self.compute_new_global_model()
        
        if self.first_round:
            self.first_round = False
            return self.k, new_global_model
        
        return 1, new_global_model
    
class DataQualityAwareFedFA(AbstractStrategy):
    def __init__(self, k: int, decay_factor: float = 0.9):
        self.k = k
        self.decay_factor = decay_factor
        self.info_queue = []
        self.params_queue = []
        self.first_round = True

    def compute_new_global_model(self) -> OrderedDict:
        mean_params = OrderedDict()
        current_time = max([float(info['timestamp']) for info in self.info_queue])
        
        quality_columns = ['label_diversity', 'spatial_diversity', 'temporal_diversity', 'sampling_regularity_std']
        quality_data = {}
        for i, info in enumerate(self.info_queue):
            info_data = {col: info[col] for col in quality_columns}
            quality_data[i] = info_data
        df = pd.DataFrame.from_dict(quality_data, orient='index')
        df_norm = (df - df.min()) / (df.max() - df.min() + 1e-8)
        df["score"] = ( 0.25 * df_norm["label_diversity"] +
                0.25 * df_norm["spatial_diversity"] +
                0.25 * df_norm["temporal_diversity"] +
                0.25 * df_norm["sampling_regularity_std"])

        total_weighted_samples = 0
        weights = []
        for i, info in enumerate(self.info_queue):
            time_weight = self.decay_factor ** (current_time - float(info['timestamp']))
            sample_weight = info['num_samples']
            quality_weight = df["score"].iloc[i]

            combined_weight = time_weight * sample_weight * quality_weight
            weights.append(combined_weight)
            total_weighted_samples += combined_weight
        
        # Normalize weights
        weights = [w / total_weighted_samples for w in weights]
        
        for key in self.params_queue[0].keys():
            weighted_sum = np.zeros_like(self.params_queue[0][key])
            for i, state_dict in enumerate(self.params_queue):
                weighted_sum += weights[i] * state_dict[key]
            mean_params[key] = weighted_sum
        
        return mean_params
    
    def get_number_of_initial_client_samples(self) -> int:
        return self.k
    
    def aggregate(self, state_dict: OrderedDict, training_info: dict) -> List:
        logging.info(f"Training info: {training_info}")

        self.info_queue.append(training_info)
        self.params_queue.append(state_dict)

        if len(self.params_queue) < self.k:
            logging.debug("Queue not full yet")
            return None, None

        if len(self.params_queue) > self.k:
            self.info_queue.pop(0)
            self.params_queue.pop(0)

        logging.debug("Queue full, computing new global model")
        new_global_model = self.compute_new_global_model()
        
        if self.first_round:
            self.first_round = False
            return self.k, new_global_model
        
        return 1, new_global_model