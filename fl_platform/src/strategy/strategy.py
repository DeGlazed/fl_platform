from typing import List, OrderedDict, Tuple
import numpy as np

class AbstractStrategy() :
    def __init__(self) -> None:
        pass
     
    def get_number_of_initial_client_samples(self) -> int:
        raise NotImplementedError("get_initial_situation() must be implemented in the subclass")
    
    def aggregate(self, local_model: List) -> List:
        raise NotImplementedError("aggregate() must be implemented in the subclass")

    def evaluate(self) -> dict:
        return None