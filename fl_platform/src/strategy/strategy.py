from typing import List, OrderedDict, Tuple

class AbstractStrategy() :
    def __init__(self) -> None:
        pass
     
    def get_initial_situation(self) -> List[str]:
        raise NotImplementedError("get_initial_situation() must be implemented in the subclass")
    
    def agregate(self, local_model: OrderedDict) -> Tuple[str, OrderedDict]:
        raise NotImplementedError("agregate() must be implemented in the subclass")