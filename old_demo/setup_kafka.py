from model import Net
import numpy as np
import torch
from collections import OrderedDict

model = Net()
init_params1 = [val.cpu().numpy().tolist() for _, val in model.state_dict().items()]
model = Net()
init_params2 = [val.cpu().numpy().tolist() for _, val in model.state_dict().items()] #this is a list

def compute_mean_params(init_params_list):
    mean_params = []
    for param_set in zip(*init_params_list):
        mean_param = np.mean(param_set, axis=0).tolist()
        mean_params.append(mean_param)
    return mean_params

# Example usage
q = [init_params1, init_params2]  # Replace with actual list of init_params
mean_params = compute_mean_params(q)
print(type(mean_params))

params_dict = zip(model.state_dict().keys(), mean_params)
state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
model.load_state_dict(state_dict, strict=True)
print(model)