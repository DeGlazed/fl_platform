from typing import List, OrderedDict, Tuple
import numpy as np

class AbstractStrategy() :
    def __init__(self) -> None:
        pass
     
    def get_number_of_initial_client_samples(self) -> int:
        raise NotImplementedError("get_initial_situation() must be implemented in the subclass")
    
    def agregate(self, local_model: List) -> List:
        raise NotImplementedError("agregate() must be implemented in the subclass")