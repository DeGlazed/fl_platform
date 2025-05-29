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

    def compute_weights(self) -> List[float]:
        total_weighted_samples = 0
        weights = []

        timestamps = [float(info['timestamp']) for info in self.info_queue]
        sample_sizes = [info['num_samples'] for info in self.info_queue]

        timestamp_min, timestamp_max = min(timestamps), max(timestamps)
        sample_min, sample_max = min(sample_sizes), max(sample_sizes)

        norm_timestamps = [(t - timestamp_min) / (timestamp_max - timestamp_min + 1e-8) for t in timestamps]
        norm_samples = [(s - sample_min) / (sample_max - sample_min + 1e-8) for s in sample_sizes]
        
        for i in range(len(self.info_queue)):
            weighted_avg = 0.5*norm_timestamps[i] + 0.5*norm_samples[i]
            weights.append(weighted_avg)
            total_weighted_samples += weighted_avg
        
        # Normalize weights
        weights = [w / total_weighted_samples for w in weights]
        return weights

    def compute_new_global_model(self) -> OrderedDict:
        mean_params = OrderedDict()

        weights = self.compute_weights()

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

    def compute_weights(self) -> List[float]:
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

        timestamps = [float(info['timestamp']) for info in self.info_queue]
        sample_sizes = [info['num_samples'] for info in self.info_queue]

        timestamp_min, timestamp_max = min(timestamps), max(timestamps)
        sample_min, sample_max = min(sample_sizes), max(sample_sizes)

        norm_timestamps = [(t - timestamp_min) / (timestamp_max - timestamp_min + 1e-8) for t in timestamps]
        norm_samples = [(s - sample_min) / (sample_max - sample_min + 1e-8) for s in sample_sizes]
        quality_weights = [df["score"].iloc[i] for i in range(len(self.info_queue))]

        for i in range(len(self.info_queue)):
            weighted_avg = 0.33*norm_timestamps[i] + 0.33*norm_samples[i] + 0.33*quality_weights[i]
            weights.append(weighted_avg)
            total_weighted_samples += weighted_avg
        
        # Normalize weights
        weights = [w / total_weighted_samples for w in weights]
        return weights

    def compute_new_global_model(self) -> OrderedDict:
        mean_params = OrderedDict()
        
        weights = self.compute_weights()
        
        for key in self.params_queue[0].keys():
            weighted_sum = np.zeros_like(self.params_queue[0][key])
            for i, state_dict in enumerate(self.params_queue):
                weighted_sum += weights[i] * state_dict[key]
            mean_params[key] = weighted_sum
        
        return mean_params
    
    def evaluate(self):
        if len(self.info_queue) < self.k:
            logging.debug("Eval queue not full yet")
            return None
        
        asyn_eval_acc = 0.0
        asyn_eval_loss = 0.0

        accs = [float(info['accuracy']) for info in self.info_queue]
        losses = [float(info['loss']) for info in self.info_queue]

        weights = self.compute_weights()

        for i in range(len(accs)):
            asyn_eval_acc += weights[i] * accs[i]
            asyn_eval_loss += weights[i] * losses[i]

        return { "accuracy": asyn_eval_acc, "loss": asyn_eval_loss }

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